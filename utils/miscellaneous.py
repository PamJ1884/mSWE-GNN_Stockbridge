try:
    import pandas as pd
except ImportError:  # only needed for CSV/DataFrame helpers (get_numerical_times, plot_line_with_deviation, ...)
    pd = None
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch_geometric.utils import scatter
try:
    from wandb import Config
except ImportError:  # only needed for wandb sweeps (fix_dict_in_config)
    Config = None

from database.graph_creation import MultiscaleMesh
from utils.dataset import to_temporal_dataset, get_inflow_volume, create_scale_mask
from training.loss import conservation_loss, mask_on_water, get_mean_error
from models.gnn import GNN, MSGNN

NUM_WATER_VARS = 2  # water depth and discharge

def get_model(model_name):
    models_dict = {'GNN': GNN,
                   'MSGNN': MSGNN}
    return models_dict[model_name]

def get_time_vector(total_time_steps, temporal_res):
    '''Returns array with temporal stamps (from 0 to total_time in hours)'''
    total_hours = total_time_steps*temporal_res/60
    time_vector = np.linspace(0, total_hours, total_time_steps+1)
    return time_vector

def add_null_time_start(time_start, temporal_array):
    '''Adds null values to the beginning of a temporal array
    
    time_start: int
        initial time of the simulation = number of null values to add
    temporal_array: np.array
        temporal array with dimensions [N_datasets, T] or [T]
    '''
    if temporal_array.ndim == 1: # [T]
        new_temporal_array = np.concatenate((np.nan*np.empty(time_start+1), temporal_array))
    elif temporal_array.ndim == 2: # [N_datasets, T]
        new_temporal_array = np.concatenate((np.nan*np.empty((
            temporal_array.shape[0], time_start+1)), temporal_array), axis=1)
    else:
        raise ValueError("Wrong temporal array dimensions")

    return new_temporal_array

def get_velocity(discharge, water_depth, epsilon=0.01):
    velocity = discharge/water_depth
    low_water = water_depth<=epsilon
    velocity[low_water] = 0
    return velocity

def get_Froude(velocity, water_depth):
    g = 9.81
    froude = velocity / torch.sqrt(g*water_depth)
    froude[water_depth <= 0] = 0
    return froude

def WD_to_FAT(WD, temporal_res, water_threshold=0, time_start=0):
    '''
    Creates flood arrival times map from temporal sequence of WD maps
    '''
    assert WD.dim() == 2, "WD must be a tensor of dimension [N, T]"
    total_time = time_start + WD.shape[-1]
    
    flooded_areas = WD > water_threshold
    flooded_time = flooded_areas.sum(1)
    FAT = -(flooded_time - total_time)
    FAT_hours = FAT*temporal_res/60
        
    return FAT_hours

def get_numerical_times(dataset_name, dataset_size, temporal_res, maximum_time,
                        overview_file='database/raw_datasets/overview.csv',
                        **temporal_test_dataset_parameters):

    time_start = temporal_test_dataset_parameters['time_start']
    time_stop = temporal_test_dataset_parameters['time_stop']

    final_time = time_stop%maximum_time + (time_stop==-1)
    
    assert final_time != -1, "I'm not sure how to interpret final_time value of -1"

    numerical_simulation_overview = pd.read_csv(overview_file, sep=',')

    mesh_dataset_train_id = numerical_simulation_overview['seed'].isin(np.arange(1,80))
    mesh_dataset_test_id = numerical_simulation_overview['seed'].isin(np.arange(81,101))

    dijk15_train_id = numerical_simulation_overview['seed'].isin([101])
    dijk15_test_id = numerical_simulation_overview['seed'].isin(np.arange(102,112))

    dataset_ids = {'mesh_dataset_train': mesh_dataset_train_id,
                   'mesh_dataset_test': mesh_dataset_test_id,
                   'multiscale_mesh_dataset_train': mesh_dataset_train_id,
                   'multiscale_mesh_dataset_test': mesh_dataset_test_id,
                   'dijkring_15_train': dijk15_train_id,
                   'dijkring_15_test': dijk15_test_id,
                   }

    ids = dataset_ids.get(dataset_name)
    if ids is None:
        raise ValueError("Wrong 'dataset_name' or maybe folder")

    computation_time = numerical_simulation_overview.loc[ids]['computation_time[s]']

    simulated_times = numerical_simulation_overview.loc[ids]['simulation_time[h]']
    model_simulated_times = (final_time-time_start)*temporal_res/60

    time_ratio = model_simulated_times/simulated_times

    return (computation_time*time_ratio).iloc[:dataset_size]

def get_speed_up(numerical_times, model_times):
    '''Calculate speed up as ratio between simulation time and DL model time'''
    speed_up = numerical_times/model_times

    return speed_up.mean(), speed_up.std()

def get_mass_conservation_loss(rollout, data):
    '''Calculates mass conservation loss for a given simulation'''
    data.area = data.area[data.node_ptr[0]:data.node_ptr[1]]
    return torch.stack([conservation_loss(rollout[:,0::NUM_WATER_VARS,t], rollout[:,0::NUM_WATER_VARS,t-1], data,
                                          (data.BC[:,t] + data.BC[:,t+1])/2) 
                                          for t in range(1, rollout.shape[-1])])

def get_binary_rollouts(predicted_rollout, real_rollout, water_threshold=0):
    '''Converts flood simulation into a binary map (1=flood, 0=no flood) for classification purposes
    ------
    water_threshold: float
        Threshold for the binary map creation, i.e., 'flood' if WD>threshold
    '''
    if predicted_rollout.dim() == 4:
        predicted_rollout_flood = predicted_rollout[:,:,0,:]>water_threshold
        real_roll_flood = real_rollout[:,:,0,:]>water_threshold
    elif predicted_rollout.dim() == 3:
        predicted_rollout_flood = predicted_rollout[:,0,:]>water_threshold
        real_roll_flood = real_rollout[:,0,:]>water_threshold

    return predicted_rollout_flood, real_roll_flood

def get_rollout_confusion_matrix(predicted_rollout, real_rollout, water_threshold=0):
    predicted_rollout_flood, real_roll_flood = get_binary_rollouts(predicted_rollout, real_rollout, water_threshold=water_threshold)

    if predicted_rollout.dim() == 4:
        nodes_dim = 1
    elif predicted_rollout.dim() == 3:
        nodes_dim = 0

    TP = (predicted_rollout_flood & real_roll_flood).sum(nodes_dim) #true positive
    TN = (~predicted_rollout_flood & ~real_roll_flood).sum(nodes_dim) #true negative
    FP = (predicted_rollout_flood & ~real_roll_flood).sum(nodes_dim) #false positive
    FN = (~predicted_rollout_flood & real_roll_flood).sum(nodes_dim) #false negative

    return TP, TN, FP, FN

def get_CSI(predicted_rollout, real_rollout, water_threshold=0):
    '''Returns the Critical Success Index (CSI) in time for a given water_threshold'''
    TP, TN, FP, FN = get_rollout_confusion_matrix(predicted_rollout, real_rollout, water_threshold=water_threshold)

    CSI = TP / (TP + FN + FP)
    # CSI[torch.isnan(CSI)] = 0

    return CSI

def get_F1(predicted_rollout, real_rollout, water_threshold=0):
    '''Returns the Critical Success Index (CSI) in time for a given water_threshold'''
    TP, TN, FP, FN = get_rollout_confusion_matrix(predicted_rollout, real_rollout, water_threshold=water_threshold)

    F1 = TP / (TP + 0.5*(FN + FP))
    # F1[torch.isnan(F1)] = 0

    return F1

def get_masked_diff(diff_roll, where_water):
    masked_diff = torch.stack([diff_roll[:,water_variable,:][where_water] 
                                    for water_variable in range(diff_roll.shape[1])])
                                
    return masked_diff

def get_rollout_loss(predicted_rollout, real_rollout, type_loss='RMSE', only_where_water=False):
    diff_roll = predicted_rollout - real_rollout

    if diff_roll.dim() == 4: #multiple simulations
        nodes_dim = 1
        water_axis = 2
    elif diff_roll.dim() == 3: #single simulation
        nodes_dim = 0
        water_axis = 1

    if only_where_water:
        where_water = mask_on_water(diff_roll, water_axis=water_axis)
        
        if diff_roll.dim() == 4:
            roll_loss = torch.stack([get_mean_error(
                get_masked_diff(diff_roll[id_dataset], where_water[id_dataset]), 
                type_loss, nodes_dim=-1) for id_dataset in range(diff_roll.shape[0])])
        elif diff_roll.dim() == 3:
            roll_loss = get_mean_error(get_masked_diff(diff_roll, where_water), type_loss, nodes_dim=-1)
    else:
        roll_loss = get_mean_error(diff_roll, type_loss, nodes_dim=nodes_dim).mean(-1)
    
    return roll_loss

def plot_line_with_deviation(time_vector, variable, with_minmax=False, ax=None, **plt_kwargs):
    '''Plots a line with mean and standard deviation
    ------
    time_vector: np.array
        vector with time stamps [h]
    variable: np.array
        vector with the variable to plot
    with_minmax: bool
        If True, also plots the envelope of the minimum and maximum values
    '''
    ax = ax or plt.gca()

    df = pd.DataFrame(np.vstack((time_vector, variable))).T
    df = df.rename(columns={0: "time"})
    df = df.set_index('time')

    mean = df.mean(1)
    std = df.std(1)
    under_line = (mean - std)
    over_line = (mean + std)

    p = ax.plot(mean, linewidth=2, marker='o', **plt_kwargs)
    color = p[0].get_color()
    ax.fill_between(std.index, under_line, over_line, color=color, alpha=.3)
    if with_minmax:
        ax.plot(df.min(1), color=color, linestyle='--', alpha=.5)
        ax.plot(df.max(1), color=color, linestyle='--', alpha=.5)
    return p

def fix_dict_in_config(wandb):
    config = dict(wandb.config)
    for k, v in config.copy().items():
        if '.' in k:
            new_key = k.split('.')[0]
            inner_key = k.split('.')[1]
            if new_key not in config.keys():
                config[new_key] = {}
            config[new_key].update({inner_key: v})
            del config[k]
    
    wandb.config = Config()
    for k, v in config.items():
        wandb.config[k] = v

def get_pareto_front(df, objective_function1, objective_function2, ascending=False):
    """Returns the pareto front of a dataframe with two objective functions
    
    df: pd.DataFrame
        Dataframe with the objective functions
    objective_function1: str
        Name of the first objective function
    objective_function2: str
        Name of the second objective function
    ascending: bool
        If True, the pareto front is calculated for ascending values of the objective functions
    """
    sorted_df = df.sort_values(by=[objective_function1, objective_function2], ascending=ascending)[[objective_function1, objective_function2]]

    pareto_front = sorted_df.values[0].reshape(1,-1)
    for var1, var2 in sorted_df.values[1:]:
        if var2 >= pareto_front[-1,1]:
            pareto_front = np.concatenate((pareto_front, np.array([[var1, var2]])), axis=0)
    
    return pareto_front

def get_sufficient_k_hops(edge_index, WD, cover_percentage=0.999):
    """Determine how many K-hops are needed to cover the cells changed in one time step

    cover_percentage: float
        Percentage of cells that need to be covered by the GNN k-hops
    """
    assert WD.dim() == 2, "The input WD matrix should contain the full original simulation [NxT]"

    row = edge_index[0]
    col = edge_index[1]

    num_nodes = WD.shape[0]
    time_steps = WD.shape[1]
    water_t1 = torch.stack([WD[:,t]>0 for t in range(1,time_steps)]).T

    fake_water = torch.zeros_like(WD)
    fake_water[WD>0] = 1
    fake_water = fake_water[:,:-1]

    changes_fully_covered = (fake_water[water_t1]>0).all()

    k=0
    while not changes_fully_covered:
        fake_water = scatter(fake_water[row], col, reduce='sum', dim=0, dim_size=num_nodes) + fake_water
        fake_water[fake_water>0] = 1
        changes_fully_covered = (fake_water[water_t1]>0).all()
        if cover_percentage < 1:
            changes_fully_covered = (fake_water[water_t1]>0).sum() > cover_percentage*water_t1.sum()
        else:
            changes_fully_covered = (fake_water[water_t1]>0).all()
        k+=1
        if k>50:
            print('This number of k-hops is probably wrong')
            break

    return k

def get_sufficient_k_hops_per_scale(edge_index, WD, edge_ptr, node_ptr, cover_percentage=0.999):
    """Determine how many K-hops are needed to cover the cells changed in one time step at the different scales"""
    
    khop_per_scale = [get_sufficient_k_hops(edge_index[:,edge_ptr[i]:edge_ptr[i+1]]-node_ptr[i], 
                                            WD[node_ptr[i]:node_ptr[i+1]], cover_percentage)
                                            for i in range(len(node_ptr)-1)]
    return khop_per_scale

class SpatialAnalysis():
    def __init__(self, predicted_rollout, prediction_times, dataset, **temporal_test_dataset_parameters):
        self.dataset = [data.cpu() for data in dataset]
        self.time_start = temporal_test_dataset_parameters['time_start']
        self.time_stop = temporal_test_dataset_parameters['time_stop']
        self.temporal_res = dataset[0].temporal_res
        self.DEMs = self._get_DEMS(self.dataset)
        temporal_dataset = to_temporal_dataset(dataset, rollout_steps=-1, **temporal_test_dataset_parameters)
        self.real_rollout = [data.y for data in temporal_dataset]
        self.predicted_rollout = predicted_rollout
        self.prediction_times = prediction_times
        if isinstance(self.dataset[0].mesh, MultiscaleMesh):
            self.BCs = torch.stack([data.BC[0] for data in self.dataset]) # only finest scale
            masks = [create_scale_mask(data.num_nodes, data.mesh.num_meshes, data.node_ptr, data) == 0 for data in self.dataset]

            self.real_rollout = [real[masks[i]] for i, real in enumerate(self.real_rollout)]
            self.predicted_rollout = [pred[masks[i]] for i, pred in enumerate(self.predicted_rollout)]
        else:
            self.BCs = torch.cat([data.BC for data in self.dataset])
        self.type_BCs = [data.type_BC.item() for data in self.dataset]
        total_time_steps = self.real_rollout[0].shape[-1]+self.time_start
        self.time_vector = get_time_vector(total_time_steps, self.temporal_res)
        
    def _get_DEMS(self, dataset):
        if isinstance(dataset, list):
            DEMs = [data.DEM for data in dataset]
        else:
            DEMs = dataset.DEM
        return DEMs

    def _plot_metric_rollouts(self, metric_name, metric_function, water_thresholds=[0.05, 0.3], ax=None):
        '''Plots metric in time for different water_thresholds
        -------
        metric_function: 
            options: get_CSI, get_F1
        '''
        if ax is None: fig, ax = plt.subplots(figsize=(7,5))

        all_metric = []
        for wt in water_thresholds:
            metric = torch.stack([metric_function(pred, real, water_threshold=wt) 
                                  for pred, real in zip(self.predicted_rollout, self.real_rollout)]).to('cpu').numpy()
            all_metric.append(metric)
            metric = add_null_time_start(self.time_start, metric)
            plot_line_with_deviation(self.time_vector, metric, ax=ax, label=f'{metric_name}$_{{{wt}}}$')
            # plt.legend()
            
        ax.set_xlabel('Time [h]')
        ax.set_title(f'{metric_name} score')
        ax.set_ylim(0,1)
        ax.grid()
        ax.legend(loc=4)
        
        return ax, np.array(all_metric)
    
    def _plot_rollouts(self, type_loss, ax=None):
        '''Plots loss in time for the different water variables'''
        if ax is None: fig, ax = plt.subplots(figsize=(7,5))

        water_labels = ['h [m]', '|q| [$m^2$/s]']
        var_colors = ['royalblue', 'purple']
        lines = []

        ax2 = ax.twinx()
        axx = ax

        diff_rollout = torch.stack([get_mean_error(pred - real, type_loss, nodes_dim=0) for pred, real 
                     in zip(self.predicted_rollout, self.real_rollout)])

        for var in range(diff_rollout.shape[1]):
            average_diff_t = diff_rollout[:,var,:].to('cpu').numpy()
            average_diff_t = add_null_time_start(self.time_start, average_diff_t)
            lines.append(plot_line_with_deviation(self.time_vector, average_diff_t, ax=axx,
                                        label=water_labels[var], c=var_colors[var])[0])
            axx = ax2

        ax.tick_params(axis='y', colors='royalblue')
        ax2.tick_params(axis='y', colors=lines[var].get_color())            
        axx = ax
        ax.set_xlabel('Time [h]')
        ax.set_title(type_loss)

        labs = [l.get_label() for l in lines]
        ax.legend(lines, labs, loc=2)
        
        return ax
    
    def _plot_BCs(self, ax=None):
        '''Plots boundary conditions in time'''
        if ax is None: fig, ax = plt.subplots(figsize=(7,5))
        
        type_BC_dict = {
            1: 'Water depth [m]',
            2: 'Discharge [$m^3$/s]'
        }
        
        if isinstance(self.dataset[0].mesh, MultiscaleMesh):
            cell_length = torch.tensor([data.edge_BC_length[0].cpu() for data in self.dataset]) # only finest scale
        else:
            cell_length = torch.cat([data.edge_BC_length.cpu() for data in self.dataset])
        inflow_nodes = (self.BCs.T * cell_length).T

        time_vector = get_time_vector(inflow_nodes.shape[-1]-1, self.temporal_res)
        plot_line_with_deviation(time_vector, inflow_nodes.cpu(), with_minmax=True, label=f'{type_BC_dict[self.type_BCs[0]]}')
        
        ax.set_xlabel('Time [h]')
        ax.set_ylabel(f'{type_BC_dict[self.type_BCs[0]]}')
        ax.set_title('Boundary conditions')
        ax.grid()
        # ax.legend(loc=1)
        
        return ax

    def _get_CSI(self, water_threshold=0):
        CSI = [get_CSI(pred, real, water_threshold=water_threshold) for pred, real in zip(self.predicted_rollout, self.real_rollout)]
        return torch.stack(CSI)
        
    def _get_F1(self, water_threshold=0):
        F1 = [get_F1(pred, real, water_threshold=water_threshold) for pred, real in zip(self.predicted_rollout, self.real_rollout)]
        return torch.stack(F1)

    def plot_CSI_rollouts(self, water_thresholds=[0.05, 0.3], ax=None):
        return self._plot_metric_rollouts('CSI', get_CSI, water_thresholds=water_thresholds, ax=ax)

    def plot_F1_rollouts(self, water_thresholds=[0.05, 0.3], ax=None):
        return self._plot_metric_rollouts('F1', get_F1, water_thresholds=water_thresholds, ax=ax)
    
    def _get_mass_loss_in_time(self):
        mass_loss = torch.stack([get_mass_conservation_loss(self.predicted_rollout[i], data.cpu()) 
                                 for i, data in enumerate(self.dataset)])*1e6 # denormalize
        return mass_loss
    
    def plot_mass_conservation(self):
        '''Plots mass conservation in time for one or more simulations'''
        fig, ax = plt.subplots(figsize=(7,5))

        mass_loss = self._get_mass_loss_in_time().cpu().numpy()
        mass_loss = add_null_time_start(self.time_start+1, mass_loss)

        inflow_volume = torch.tensor([[get_inflow_volume(data, data.BC[:,t:t+2].mean(1)) 
                                                for t in range(mass_loss.shape[1]-1)] 
                                                for data in self.dataset])

        inflow_volume = add_null_time_start(self.time_start, inflow_volume)
        # cum_mass_loss = np.nancumsum(mass_loss, axis=-1)
        # cum_mass_loss_norm = cum_mass_loss/np.nancumsum(inflow_volume.mean(0), axis=-1)

        plot_line_with_deviation(self.time_vector, inflow_volume, label='Inflow volume', ax=ax)
        plot_line_with_deviation(self.time_vector, mass_loss, label='Error', ax=ax)

        ax.legend()
        ax.set_xlabel('Time [h]')
        ax.set_ylabel('Volume per time step [$m^3$]')

        plt.tight_layout()
        
        return mass_loss

    def _get_rollout_loss(self, type_loss='RMSE', only_where_water=False):
        rollout_losses = [get_rollout_loss(pred, real, type_loss=type_loss, only_where_water=only_where_water) 
                          for pred, real in zip(self.predicted_rollout, self.real_rollout)]

        return torch.stack(rollout_losses)

    def plot_loss_per_simulation(self, type_loss='RMSE', water_thresholds=[0.05, 0.3], 
                                 ranking='loss', only_where_water=False, figsize=(20,12)):
        '''Plot sorted loss for each simulation in a dataset
        ranking: criterion to sort simulations
            options: 'loss', 'CSI'
        '''
        rollout_loss = self._get_rollout_loss(type_loss=type_loss, only_where_water=only_where_water)
        CSIs = torch.stack([self._get_CSI(wt) for wt in water_thresholds], 1).nanmean(2).to('cpu')

        assert rollout_loss.dim() == 2, "rollout_loss should have dimension [S, O]"\
            "where S is the number of simulations and O is the output dimension"
        if rollout_loss.shape[0] == 1:
            raise ValueError("This plot works only for multiple simulations")
        if isinstance(rollout_loss, torch.Tensor):
            rollout_loss = rollout_loss.to('cpu').numpy()

        fig, axs = plt.subplots(3, 1, figsize=figsize, sharex='col')

        if ranking == 'loss':
            sorted_ids = rollout_loss.mean(1).argsort()
        elif ranking == 'CSI':
            sorted_ids = CSIs.mean(1).argsort().flip(-1).numpy()
        else:
            raise ValueError("ranking can only be either 'loss' or 'CSI'")

        positions=range(len(sorted_ids))
        
        water_variables = rollout_loss.shape[1]
        if water_variables == 1:
            water_labels = ['h [m]']
            var_colors = ['royalblue']
        elif water_variables == 2:
            water_labels = ['h [m]', '|q| [$m^2$/s]']
            var_colors = ['royalblue', 'purple']
        elif water_variables == 3:
            water_labels = ['h [m]', 'qx [$m^2$/s]', 'qy [$m^2$/s]']
            var_colors = ['royalblue', 'orange', 'darkgreen']
            
        axs[0].set_title(f'{ranking} ranking for test simulations')
        n_x_ticks = range(len(sorted_ids))
        axs[0].boxplot([self.DEMs[i] for i in sorted_ids], positions=positions)
        axs[0].set_ylabel(r'DEM [m]')

        for i, (color, label) in enumerate(zip(var_colors, water_labels)):
            axs[1].plot(rollout_loss[sorted_ids, i], 'o--', label=label, c=color)
        axs[1].set_ylabel(type_loss)
        axs[1].set_yscale('log')
        axs[1].legend()
        
        axs[2].set_xticks(n_x_ticks)
        axs[2].set_xticklabels(sorted_ids)
        
        [axs[2].plot(CSIs[sorted_ids, i], 'o--', label=f'CSI$_{{{wt}}}$') for i, wt in enumerate(water_thresholds)]
        axs[2].set_ylim(0,1)
        axs[2].set_xlabel('Simulation id')
        axs[2].set_ylabel('CSI')
        axs[2].legend()

        fig.subplots_adjust(wspace=0, hspace=0.05)

        return sorted_ids

    def plot_summary(self, numerical_times, type_loss='RMSE', water_thresholds=[0.05, 0.3], 
                     only_where_water=False, figsize=(10,5)):

        fig, axs = plt.subplots(1, 3, figsize=figsize)

        RMSE = self._get_rollout_loss(type_loss=type_loss, only_where_water=only_where_water).cpu()
        CSIs = [self._get_CSI(wt).nanmean(1).cpu() for wt in water_thresholds]

        axs[0].boxplot(CSIs)
        axs[0].set_ylim(0,1)
        axs[0].set_xticklabels([r'$\tau$'f'={wt}m' for wt in water_thresholds])
        axs[0].set_title(r'CSI$_\tau$ [-]')

        axs[1].boxplot((RMSE[:,0], RMSE[:,1:].mean(1)))
        axs[1].set_xticklabels(('h [m]', '|q| [$m^2$/s]'))
        axs[1].set_title(f'{type_loss}')
        axs[1].set_yscale('log')

        axs[2].boxplot((self.prediction_times, numerical_times))
        axs[2].set_title('Execution times [sec]')
        axs[2].set_xticklabels(('DL', 'Numerical'))
        axs[2].set_ylim(0)

        plt.tight_layout()

        return fig