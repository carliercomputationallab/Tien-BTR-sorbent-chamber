import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import optimize, interpolate
import openpyxl
import csv
from matplotlib.lines import Line2D

#%%

def objective(x0, c_ddr, data_ds, data_w, f_avg, substance, t_end):
    V_waste = 0 #ml
    c_w = pd.DataFrame({'waste-sim': [0]*t_end}) #mg/ml
    # s = [data_ds.loc[0,substance]] #mg/ml
    s = [0] #mg/ml
    k_Th = x0[0] #ml/mg.min
    q_e = x0[1] #mg/g
    solute_mass = np.zeros(t_end) #mg

    for t in range(1, t_end):
        c_ds = c_ddr*(1/(1+np.exp(k_Th*q_e*x_AC/f_avg-k_Th*c_ddr*t)))
        solute_mass[t] = (c_ddr - c_ds) * f_avg * 1 #+solute_mass[t-1] --- uncomment for cumulative
        s.append(c_ds) 
        c_w.loc[t] = (f_avg*c_ds*1+V_waste*c_w.loc[t-1])/(f_avg + V_waste) 
        V_waste += f_avg * 1 
    return c_w, s, solute_mass

def optimise_fn(x0, c_ddr, data_ds, data_w, f_avg, substance):
    # print(data_ds[substance], data_w[substance], f_avg)
    V_waste = 0 
    c_w = pd.DataFrame({'waste': [0]*t_end}) 
    s = pd.DataFrame({'sorbents': [0]*t_end}) 
    # s.loc[0] = data_ds.loc[0,substance]
    k_Th = x0[0] 
    q_e = x0[1] 
    
    for t in range(1, t_end):
        c_ds = c_ddr*(1/(1+np.exp(k_Th*q_e*x_AC/f_avg-k_Th*c_ddr*t))) 
        s.loc[t] = c_ds
        c_w.loc[t] = (f_avg*c_ds*1+V_waste*c_w.loc[t-1])/(f_avg + V_waste) 
        V_waste += f_avg * 1 
    # print(sum(np.sqrt((data_ds[substance].astype(float)-s.loc[data_ds.index, 'sorbents'])**2)), sum(np.sqrt((data_w[substance].astype(float)-c_w.loc[data_w.index, 'waste'])**2)))
    return np.sqrt(sum((data_ds[substance].astype(float)-s.loc[data_ds.index, 'sorbents'])**2)) +np.sqrt(sum((data_w[substance].astype(float)-c_w.loc[data_w.index, 'waste'])**2))

#read flow rate directly from the file
def read_excel_cell(file_path, sheet_name, cell_address):
    workbook = openpyxl.load_workbook(file_path, data_only=True)
    sheet = workbook[sheet_name]
    cell_value = sheet[cell_address].value
    workbook.close()
    return cell_value
#%%
file_path = 'Data.xlsx'
sheet_name = ['Experiment 0', 'Experiment 1', 'Experiment 2', 'Experiment 3', 'Experiment 5'] 
data = {}

for sheet_name in sheet_name:
    # cols_name = ['Time', 'Bicarbonate', 'Magnesium', 'Sodium', 'Calcium', 'Glucose']
    cols_name = ['Time', 'Bicarbonate', 'Magnesium', 'Sodium', 'Calcium', 'Glucose']
    # Downstream sorbent
    df = pd.read_excel(file_path, sheet_name=sheet_name, header = 0)
    df = df.T
    data_ds = df[df.iloc[:,1] == 'line'].reset_index(drop = True).dropna(axis = 1, how = 'all').drop(columns=[0, 1, 3])
    
    
    data_ds.columns = cols_name
    data_ds['Time'] = data_ds['Time'].str.replace('t=', '').astype(int)
    data_ds.set_index(['Time'], inplace = True)
    
    # Waste
    data_w = df[df.iloc[:,1] == 'reservoir'].reset_index(drop = True).dropna(axis = 1, how = 'all').drop(columns=[0, 1, 3])
    data_w.columns = cols_name
    data_w['Time'] = data_w['Time'].str.replace('t=', '').astype(int) 
    data_w.set_index(['Time'], inplace = True)
    
    #starting concentration
    c_ddr = df.iloc[0:2].copy(deep = True) 
    c_ddr.reset_index(inplace = True)
    c_ddr.columns = c_ddr.iloc[0]
    c_ddr = c_ddr.loc[1, cols_name[1:]]
    
    
    #add the cocnentrations at time 0
    new_row = pd.DataFrame([[21, 0, 132, 0, 0]], columns=data_ds.columns) #preconditioning liquid
    data_ds = pd.concat([new_row, data_ds])
    new_row = pd.DataFrame([[0, 0, 0, 0, 0]], columns=data_ds.columns)
    data_w =  pd.concat([new_row, data_w])
    
    #average flow rate
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    f_avg = df[df.iloc[:,0] == 'average flow rate'].iloc[0,1]
    
    
    
    # print(c_ddr)
    data[sheet_name] = {'Downstream sorbent': data_ds, 'Waste': data_w, 'Flow rate':f_avg, 'Initial conc': c_ddr}

    # print(f"Sheet: {sheet_name}")
    # print("Downstream sorbent concentration:")
    # print(data_ds)
    # print("Waste concentration:")
    # print(data_w)

molecular_weight = {'Bicarbonate': 61.0168, 
                    'Magnesium': 24.3050, 
                    'Sodium': 22.9898, 
                    'Calcium': 40.0784, 
                    'Glucose': 180.156
                    }



x_AC = 300 #g
output_data = {}
#%%
for sheet_name, data_dict in data.items():    
    
    # simulations = pd.DataFrame(columns = ['Solute', 'k_Th', 'q_e'])
    simulations = []
    
    for substance in cols_name[1:]: # it will automatically rotate through all substances for one sheet_name
        data_ds = data_dict['Downstream sorbent'].astype(float, errors = 'ignore')
        data_ds[substance] = data_ds[substance] * molecular_weight[substance] / 1000
        data_w = data_dict['Waste'].astype(float, errors = 'ignore')
        data_w[substance] = data_w[substance] * molecular_weight[substance] / 1000
        # print(f"Sheet: {sheet_name}")
        # print(data_ds[substance], data_w[substance])
        c_ddr = data_dict['Initial conc'][substance]* molecular_weight[substance] / 1000
        time = data_ds.index
        f_avg = data_dict['Flow rate']
        t_end = data_ds.index[-1]+1
        
        
        # initialise two lists to collect the fitted values and the objective function
        x_val = []
        obj_fn = []
        for var in range(10):
            x0 = np.random.random(2)
            # sending back data[sheet_name] to the function instead pf df
            result = scipy.optimize.minimize(optimise_fn, x0, args = (c_ddr, data_ds, data_w, f_avg, substance),
                                            method='SLSQP', bounds = [(0, np.inf) for _ in x0], options = {"maxiter" : 1000, "disp": False})
            x_val.append(result['x'])
            obj_fn.append(result['fun'])
        x_sel = x_val[np.argmin(obj_fn)]
        
        print(f"Sheet: {sheet_name}, Solute:{substance}")
        print(f"Kth:{x_sel[0]} and q_e:{x_sel[1]}")
        
        expt = pd.DataFrame({
            'Solute': [substance],
            'k_Th': [x_sel[0]],
            'q_e': [x_sel[1]]
            })
        
        # simulations=pd.concat([simulations, expt])
        simulations.extend([expt])
        output_data[sheet_name] = pd.concat(simulations, ignore_index=True)
#%%        
output_file_path = 'output.csv'
with open(output_file_path, 'w', newline='') as csvfile:
    fieldnames = ['Experiment', 'Solute', 'k_Th', 'q_e']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    
    for sheet_name, simulations in output_data.items():
        for index, row in simulations.iterrows():
            writer.writerow({
                'Experiment': sheet_name,
                'Solute': row['Solute'],
                'k_Th': row['k_Th'],
                'q_e': row['q_e']
            })
        
        simulations.set_index(['Solute'], inplace = True)
#Plot
#%%

for sheet_name, data_dict in data.items():    
    fig, axs = plt.subplots(5, 2, figsize=(12, 15))
    
    for i, substance in enumerate(cols_name[1:], 0):
        data_ds = data_dict['Downstream sorbent'].astype(float, errors = 'ignore')
        data_ds[substance] = data_ds[substance] * molecular_weight[substance] / 1000
        data_w = data_dict['Waste'].astype(float, errors = 'ignore')
        data_w[substance] = data_w[substance] * molecular_weight[substance] / 1000
        # print(f"Sheet: {sheet_name}")
        # print(data_ds[substance], data_w[substance])
        c_ddr = data_dict['Initial conc'][substance] * molecular_weight[substance] / 1000
        t_end = data_ds.index[-1]+1
        time = data_ds.index
        x_sel = output_data[sheet_name].loc[substance]
        print(x_sel)
        c_w, s, solute_mass = objective(x_sel, c_ddr, data_ds, data_w, f_avg, substance, t_end)
        axs[i, 0].scatter(time, data_w[substance], label='expt. waste', c='b')
        c_w.plot(ax = axs[i, 0], c = 'b')
        axs[i, 0].legend()
        axs[i, 0].set_title(substance)
        
        axs[i, 1].scatter(time, data_ds[substance], c='k', label='expt. sorbents')
        axs[i, 1].plot(range(0,t_end), s, c='k', label='downstream sorbent')
        axs[i, 1].legend()
        axs[i, 1].set_title(substance)
    plt.suptitle(sheet_name)
    plt.xlabel('Time')
    plt.ylabel('Solute concentration')
    plt.tight_layout()
    plt.show()
#%%
'''
Total solute removed
'''
fig, axs = plt.subplots(5, 2, figsize=(12, 15))
cs = ['#5ba300', '#89ce00', '#0073e6', '#e6308a', '#b51963']
cw = ['#c44601', '#f57600', '#8babf1', '#0073e6', '#054fb9']

j = 0

flowrate = [60, 75, 123]
# flowrate = [20, 15, 25]
TSR = []

for sheet_name, data_dict in data.items():    
    
    if sheet_name == 'Experiment 3' or sheet_name == 'Experiment 5':
        continue
    CW = cw[j]
    CS = cs[j]
    solute_removal = 0
    for i, substance in enumerate(cols_name[1:], 0):
        data_ds = data_dict['Downstream sorbent'].astype(float, errors = 'ignore')
        data_ds[substance] = data_ds[substance] * molecular_weight[substance] / 1000
        data_w = data_dict['Waste'].astype(float, errors = 'ignore')
        data_w[substance] = data_w[substance] * molecular_weight[substance] / 1000
        # print(f"Sheet: {sheet_name}")
        # print(data_ds[substance], data_w[substance])
        c_ddr = data_dict['Initial conc'][substance] * molecular_weight[substance] / 1000
        t_end = data_ds.index[-1]+1
        time = data_ds.index
        x_sel = output_data[sheet_name].loc[substance]
        
        c_w, s, solute_mass = objective(x_sel, c_ddr, data_ds, data_w, f_avg, substance, t_end)
        axs[i, 0].scatter(time, data_w[substance], label=str(flowrate[j]) + 'expt. waste', c=CW)
        solute_removal += sum(solute_mass)
        c_w.plot(ax = axs[i, 0], c = CW, legend = None)
        if i == 0:
            axs[i, 0].legend()
        
        axs[i, 0].set_title(substance)
        
        axs[i, 1].scatter(time, data_ds[substance], c=CS, label=str(flowrate[j])+ 'expt. sorbents')
        axs[i, 1].plot(range(0,t_end), s, c=CS, label=sheet_name+'sim')
        if i == 0:
            axs[i, 1].legend()
        axs[i, 1].set_title(substance)
    TSR.append(solute_removal)
    j += 1
    plt.suptitle(sheet_name)
    plt.xlabel('Time')
    plt.ylabel('Solute concentration')
    plt.tight_layout()
    plt.show()
    
plt.figure()
df_TSR = pd.DataFrame({'flowrate': flowrate, 'TSR (mg)': TSR})
df_TSR.sort_values(by = ['flowrate'], inplace = True)   
df_TSR.plot(x='flowrate', y='TSR (mg)', style='o')

#%%
'''
volume passed versus solute absorbed
'''
fig, axs = plt.subplots(1, 1, figsize=(6, 6))
cs = ['#5ba300', '#89ce00', '#0073e6', '#e6308a', '#b51963']
cw = ['#c44601', '#f57600', '#8babf1', '#0073e6', '#054fb9']

j = 0

flowrate = [60, 75, 123]
# flowrate = [20, 15, 25]
TSR = []
labels = ['q60-x20', 'q75-x20', 'q125-x20', 'q78-x15', 'q72-x25']
j = 0
for sheet_name, data_dict in data.items():    
    
    fr = data_dict['Flow rate']
    
    solute_removal = 0
    for i, substance in enumerate(cols_name[5:], 0):
        data_ds = data_dict['Downstream sorbent'].astype(float, errors = 'ignore')
        data_ds[substance] = data_ds[substance] * molecular_weight[substance] / 1000
        data_w = data_dict['Waste'].astype(float, errors = 'ignore')
        data_w[substance] = data_w[substance] * molecular_weight[substance] / 1000
        # print(f"Sheet: {sheet_name}")
        # print(data_ds[substance], data_w[substance])
        c_ddr = data_dict['Initial conc'][substance] * molecular_weight[substance] / 1000
        t_end = data_ds.index[-1]+1
        time = data_ds.index
        x_sel = output_data[sheet_name].loc[substance]
        
        c_w, s, solute_mass = objective(x_sel, c_ddr, data_ds, data_w, f_avg, substance, t_end)
        
        volume_passed = np.arange(t_end)*fr*c_ddr
        axs.scatter(volume_passed, solute_mass, label=labels[j])
    j = j + 1

    plt.xlabel('Volume passed*conc, mg')
    plt.ylabel('Solute conc effluent - influent, mg/ml')
    plt.legend()
    plt.tight_layout()
    plt.show()
    

    
#%%
'''
k_Th with Q and x0
qe with Q and x0
'''
# #Analyze how k_Th and q_e change with different flow rate and initial solute conc.
df = pd.read_csv('output.csv', delimiter=',')

# Experiments which have different flow rate
experiments = ['Experiment 0', 'Experiment 1', 'Experiment 2']

# Flow rate mapping for each experiment
flow_rate_mapping = {'Experiment 0': 60, 'Experiment 1': 100, 'Experiment 2': 150}

# Get a list of unique solutes in the DataFrame
unique_solutes = df['Solute'].unique()

# Create a figure for k_Th with subplots for each solute
fig_k_Th, axs_k_Th = plt.subplots(1, len(unique_solutes), figsize=(15, 5), sharey=False)

# Create a figure for q_e with subplots for each solute
fig_q_e, axs_q_e = plt.subplots(1, len(unique_solutes), figsize=(15, 5), sharey=False)

# Loop through each solute and plot the corresponding subplot for k_Th and q_e
for i, solute in enumerate(unique_solutes):
    # Filter the DataFrame to get the rows for the desired experiments and current solute
    experiment_solute = df[(df['Experiment'].isin(experiments)) & (df['Solute'] == solute)]
    
    # Get k_Th values for each flow rate (60, 100, 150) using the mapping
    k_Th_values = []
    q_e_values = []
    for experiment in experiments:
        flow_rate = flow_rate_mapping[experiment]
        k_Th = experiment_solute.loc[experiment_solute['Experiment'] == experiment, 'k_Th'].values[0]
        k_Th_values.append((flow_rate, k_Th))
        
        q_e = experiment_solute.loc[experiment_solute['Experiment'] == experiment, 'q_e'].values[0]
        q_e_values.append((flow_rate, q_e))
    
    # Sort the values based on flow rate to ensure the correct order in the plot
    k_Th_values.sort(key=lambda x: x[0])
    q_e_values.sort(key=lambda x: x[0])
    
    # Separate flow rates and k_Th values for the plot
    flow_rates_k_Th, k_Th = zip(*k_Th_values)
    
    # Plot the current solute in the corresponding subplot for k_Th
    axs_k_Th[i].plot(flow_rates_k_Th, k_Th, marker='o', linestyle='-')
    axs_k_Th[i].set_xlabel('Flow Rate')
    axs_k_Th[i].set_ylabel('k_Th')
    axs_k_Th[i].set_title(f'{solute}')
    axs_k_Th[i].grid(True)
    
    # Separate flow rates and q_e values for the plot
    flow_rates_q_e, q_e = zip(*q_e_values)
    
    # Plot the current solute in the corresponding subplot for q_e
    axs_q_e[i].plot(flow_rates_q_e, q_e, marker='o', linestyle='-')
    axs_q_e[i].set_xlabel('Flow Rate')
    axs_q_e[i].set_ylabel('q_e')
    axs_q_e[i].set_title(f'{solute}')
    axs_q_e[i].grid(True)

# Adjust the layout of the subplots for better spacing for k_Th figure
fig_k_Th.tight_layout()

# Save the figure for k_Th to a file
fig_k_Th.savefig('Q_k_Th_figures.png')

# Adjust the layout of the subplots for better spacing for q_e figure
fig_q_e.tight_layout()

# Save the figure for q_e to a file
fig_q_e.savefig('Q_q_e_figures.png')



#Analyze how k_Th and q_e change with different flow rate and initial solute conc.
df = pd.read_csv('output.csv', delimiter=',')

# Experiments which have different flow rate
experiments = ['Experiment 3', 'Experiment 1', 'Experiment 5']

# Flow rate mapping for each experiment
initial_conc_mapping = {'Experiment 3': 15, 'Experiment 1': 20, 'Experiment 5': 25}

# Get a list of unique solutes in the DataFrame
unique_solutes = df['Solute'].unique()

# Create a figure for k_Th with subplots for each solute
fig_k_Th, axs_k_Th = plt.subplots(1, len(unique_solutes), figsize=(15, 5), sharey=False)

# Create a figure for q_e with subplots for each solute
fig_q_e, axs_q_e = plt.subplots(1, len(unique_solutes), figsize=(15, 5), sharey=False)

# Loop through each solute and plot the corresponding subplot for k_Th and q_e
for i, solute in enumerate(unique_solutes):
    # Filter the DataFrame to get the rows for the desired experiments and current solute
    experiment_solute = df[(df['Experiment'].isin(experiments)) & (df['Solute'] == solute)]
    
    # Get k_Th values for each flow rate (60, 100, 150) using the mapping
    k_Th_values = []
    q_e_values = []
    for experiment in experiments:
        initial_conc = initial_conc_mapping[experiment]
        k_Th = experiment_solute.loc[experiment_solute['Experiment'] == experiment, 'k_Th'].values[0]
        k_Th_values.append((initial_conc, k_Th))
        
        q_e = experiment_solute.loc[experiment_solute['Experiment'] == experiment, 'q_e'].values[0]
        q_e_values.append((initial_conc, q_e))
    
    # Sort the values based on flow rate to ensure the correct order in the plot
    k_Th_values.sort(key=lambda x: x[0])
    q_e_values.sort(key=lambda x: x[0])
    
    # Separate flow rates and k_Th values for the plot
    initial_conc, k_Th = zip(*k_Th_values)
    
    # Plot the current solute in the corresponding subplot for k_Th
    axs_k_Th[i].plot(initial_conc, k_Th, marker='o', linestyle='-')
    axs_k_Th[i].set_xlabel('Initial concentration')
    axs_k_Th[i].set_ylabel('k_Th')
    axs_k_Th[i].set_title(f'{solute}')
    axs_k_Th[i].grid(True)
    
    # Separate flow rates and q_e values for the plot
    initial_conc, q_e = zip(*q_e_values)
    
    # Plot the current solute in the corresponding subplot for q_e
    axs_q_e[i].plot(initial_conc, q_e, marker='o', linestyle='-')
    axs_q_e[i].set_xlabel('Initial concentration')
    axs_q_e[i].set_ylabel('q_e')
    axs_q_e[i].set_title(f'{solute}')
    axs_q_e[i].grid(True)

# Adjust the layout of the subplots for better spacing for k_Th figure
fig_k_Th.tight_layout()

# Save the figure for k_Th to a file
fig_k_Th.savefig('x0_k_Th_figures.png')

# Adjust the layout of the subplots for better spacing for q_e figure
fig_q_e.tight_layout()

# Save the figure for q_e to a file
fig_q_e.savefig('x_0_q_e_figures.png')