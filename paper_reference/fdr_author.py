import gurobipy as grb
import numpy as np
import timeit
import scipy
import sklearn
from ucimlrepo import fetch_ucirepo 
import math



# Setting Up Data Generation    
def generate_n_sample_vec(weights,total_samples):
    n_classes = len(weights)
    n_samples_vec = np.zeros(len(weights))
    for j in range(len(n_samples_vec)):
        n_samples_vec[j] = np.round(weights[j]*total_samples)
        
    n_samples_vec = n_samples_vec.astype(int)
    
    count_add = 0
    count_subt = 0
    while sum(n_samples_vec) < total_samples:
        idx_add = count_add%n_classes
        n_samples_vec[idx_add] += 1
        count_add += 1
        
    while sum(n_samples_vec) > total_samples:
        idx_subt = count_subt%n_classes
        n_samples_vec[idx_subt] -= 1
        count_subt += 1
        
    return n_samples_vec

def normalize_data(data_in,lims):
    row,col = data_in.shape
    data_norm = np.zeros([row,col])
    for c in range(col):
        tmp = data_in[:,c]
        data_norm[:,c] = (tmp - lims[1,c])/(lims[0,c] - lims[1,c])
    
    return data_norm

def change_labels(perc_wrong_y,total_samples,y):
    n_wrong = round(perc_wrong_y*total_samples)
    idx_wrong = np.random.randint(0, high=total_samples, size=n_wrong)
    y[idx_wrong] = y[idx_wrong]*-1
    
    return y
    
def gen_data(G,perc_train,perc_test,perc_wrong_y=0,perc_g_vec=-1):
    
    # fetch dataset 
    all_data = fetch_ucirepo(id=267) 

    # data (as pandas dataframes) 
    x_all = np.asarray(all_data.data.features)
    y_all_raw = np.asarray(all_data.data.targets)
    
    y_all = np.zeros(y_all_raw.shape)
    y_all[y_all_raw == 0] = 0
    y_all[y_all_raw == 1] = 1
    y_all = y_all[:,0]

    
    n_samples,n_features = x_all.shape
    
    weights_train = [sum(y_all == 0)/n_samples,sum(y_all == 1)/n_samples]
    weights_test = [sum(y_all == 0)/n_samples,sum(y_all == 1)/n_samples]
    
    
    # Shuffle Data
    idx = np.random.permutation(n_samples)
    x_all = x_all[idx]
    y_all = y_all[idx]
    
    # Compute Training and Test Samples
    n_train = math.floor(perc_train*n_samples) - 2
    n_test = math.floor(perc_test*n_samples) - 2
    
    
    
    # Separate test set
    x_test = np.zeros([n_test,n_features])
    y_test = np.zeros(n_test)
    
    test_idx_vec = generate_n_sample_vec(weights_test,n_test)
    test_idx_vec = np.insert(test_idx_vec,0,0)
    
    x_rem = np.zeros([len(x_all) - sum(test_idx_vec),n_features])
    y_rem = np.zeros(len(y_all) - sum(test_idx_vec))
    
    prev_idx = 0
    for c in range(2):
        x_c = x_all[y_all == c,:]
        n_samples_c = test_idx_vec[c+1]
        x_test[sum(test_idx_vec[:c+1]):sum(test_idx_vec[:c+1]) + test_idx_vec[c+1],:] = x_c[:n_samples_c,:]
        y_test[sum(test_idx_vec[:c+1]):sum(test_idx_vec[:c+1]) + test_idx_vec[c+1]] = c
        
        idx_rem = len(x_c) - test_idx_vec[c+1]
        x_rem[prev_idx:len(x_c) - test_idx_vec[c+1] + prev_idx,:] = x_c[test_idx_vec[c+1]:,:]
        y_rem[prev_idx:len(x_c) - test_idx_vec[c+1] + prev_idx] = c
        
        prev_idx = len(x_c) - test_idx_vec[c+1] + prev_idx

    y_test = y_test.astype(int)
    y_test[y_test == 0] = -1
    
    
    
    # Creating Client Training Sets
    if perc_g_vec == -1:
        tmp_perc_vec = np.random.uniform(low=0.1,high=1.0,size=G)
        perc_g_vec = tmp_perc_vec/np.sum(tmp_perc_vec)
        
    elif perc_g_vec == 0:
        perc_g_vec = np.ones(G)*(1/G)
        
        
    Ng_vec = generate_n_sample_vec(perc_g_vec,n_train)
    
    x_central = np.zeros([np.sum(Ng_vec),n_features])
    x_central_norm = np.zeros([np.sum(Ng_vec),n_features])
    y_central = np.zeros(np.sum(Ng_vec))
    
    client_sets = {}
    client_sets_norm = {}
    
    x_all = x_rem
    y_all = y_rem

    # Compute Limits
    lims = np.zeros([2,n_features])
    for c in range(n_features):
        lims[0,c] = np.max(x_all[:,c])
        lims[1,c] = np.min(x_all[:,c])

    x_test_norm = normalize_data(x_test,lims)
    
    for g in range(len(Ng_vec)):
        x_train_g = np.zeros([Ng_vec[g],n_features])
        y_train_g = np.zeros(Ng_vec[g])

        train_g_idx_vec = generate_n_sample_vec(weights_train,Ng_vec[g])
        train_g_idx_vec = np.insert(train_g_idx_vec,0,0)

        x_rem = np.zeros([len(x_all) - sum(train_g_idx_vec),n_features])
        y_rem = np.zeros(len(y_all) - sum(train_g_idx_vec))
        
        prev_idx = 0
        for c in range(2):
            x_c = x_all[y_all == c,:]
            n_samples_c_g = train_g_idx_vec[c+1]
            x_train_g[sum(train_g_idx_vec[:c+1]):sum(train_g_idx_vec[:c+1]) + train_g_idx_vec[c+1],:] = x_c[:n_samples_c_g,:]
            y_train_g[sum(train_g_idx_vec[:c+1]):sum(train_g_idx_vec[:c+1]) + train_g_idx_vec[c+1]] = c
            
            idx_rem = len(x_c) - train_g_idx_vec[c+1]
            x_rem[prev_idx:len(x_c) - train_g_idx_vec[c+1] + prev_idx,:] = x_c[train_g_idx_vec[c+1]:,:]
            y_rem[prev_idx:len(x_c) - train_g_idx_vec[c+1] + prev_idx] = c
            
            prev_idx = len(x_c) - train_g_idx_vec[c+1] + prev_idx
            
        y_train_g[y_train_g == 0] = -1
        x_train_g_norm = normalize_data(x_train_g,lims)
        
        if perc_wrong_y > 0:
            y_train_g = change_labels(perc_wrong_y,len(y_train_g),y_train_g)

        client_sets['x'+str(g)] = x_train_g
        client_sets['y'+str(g)] = y_train_g

        client_sets_norm['x'+str(g)] = x_train_g_norm
        client_sets_norm['y'+str(g)] = y_train_g
        
        if g == 0:
            x_central[0:Ng_vec[g],:] = x_train_g
            x_central_norm[0:Ng_vec[g],:] = x_train_g_norm
        
            y_central[0:Ng_vec[g]] = y_train_g
        elif g > 0:
            x_central[np.sum(Ng_vec[:g]):Ng_vec[g]+np.sum(Ng_vec[:g]),:] = x_train_g
            x_central_norm[np.sum(Ng_vec[:g]):Ng_vec[g]+np.sum(Ng_vec[:g]),:] = x_train_g_norm
        
            y_central[np.sum(Ng_vec[:g]):Ng_vec[g]+np.sum(Ng_vec[:g])] = y_train_g
        
        x_all = x_rem
        y_all = y_rem
        
    central_set = {'x': x_central, 'y': y_central}
    central_set_norm = {'x': x_central_norm, 'y': y_central}
    
    test_set = {'x': x_test, 'y': y_test}
    test_set_norm = {'x': x_test_norm, 'y': y_test}
            
        
    
    return central_set, client_sets, test_set, central_set_norm, client_sets_norm, test_set_norm, Ng_vec
        
class subgradient_alg:
    """Subgradient-based algorithm for traininf distributionally robust SVMs"""
    
    def __init__(self,param,init_vars,Ng_vec):
        self.epsilon = param['epsilon']
        self.kappa = param['kappa']
        self.pnorm = param['pnorm']
        self.max_iter = param['max_iter']
        self.stepsize = param['stepsize']
        self.w_global = init_vars['w']
        self.Ng_vec = Ng_vec
        self.G = len(Ng_vec)
        self.N = np.sum(Ng_vec)
        self.alphag_vec = np.ones(np.shape(Ng_vec))*(1/self.G)
            
    def extremal_dist(self,data,g):
        
        x_train = data['x']
        y_train = data['y']

        row, col = x_train.shape
        optimal = {}
        
        # Step 0: create model
        model = grb.Model('Extremal_Distribution_SVM_'+str(g))
        model.setParam('OutputFlag', False)
        
        # Step 1: define decision variables
        var_b_pos = {}
        var_b_neg = {}
        var_q_pos = {}
        var_q_neg = {}
        slack_q_pos = {}
        slack_q_neg = {}
        aux_q_pos = {}
        aux_q_neg = {}
        slack_b_pos = {}
        slack_b_neg = {}
        
        for n in range(row):
            var_b_pos[n] = model.addVar(vtype=grb.GRB.CONTINUOUS, lb=1e-5)
            var_b_neg[n] = model.addVar(vtype=grb.GRB.CONTINUOUS, lb=1e-5)
            
        for n in range(row):
            for p in range(col):
                var_q_pos[n,p] = model.addVar(vtype=grb.GRB.CONTINUOUS, lb=-grb.GRB.INFINITY)
                var_q_neg[n,p] = model.addVar(vtype=grb.GRB.CONTINUOUS, lb=-grb.GRB.INFINITY)
                if self.pnorm == 1:
                    slack_q_pos[n,p] = model.addVar(vtype=grb.GRB.CONTINUOUS)
                    slack_q_neg[n,p] = model.addVar(vtype=grb.GRB.CONTINUOUS)
            if self.pnorm == float('Inf'):
                aux_q_pos[n] = model.addVar(vtype=grb.GRB.CONTINUOUS)
                aux_q_neg[n] = model.addVar(vtype=grb.GRB.CONTINUOUS)
                
                    
                
        # Step 2: integrate variables
        model.update()
        
        # Step3: define constraints
        if self.pnorm == 1:
            for p in range(col):
                for n in range(row):
                    model.addConstr(var_q_pos[n,p] <= slack_q_pos[n,p])
                    model.addConstr(-var_q_pos[n,p] <= slack_q_pos[n,p])
                    model.addConstr(var_q_neg[n,p] <= slack_q_neg[n,p])
                    model.addConstr(-var_q_neg[n,p] <= slack_q_neg[n,p])
            model.addConstr(grb.quicksum(\
                grb.quicksum(slack_q_pos[n,p] for p in range(col)) + grb.quicksum(slack_q_neg[n,p] for p in range(col)) +\
                self.kappa[g]*var_b_neg[n] for n in range(row)) <= row*self.epsilon[g])

        elif self.pnorm == float('Inf'):
            for p in range(col):
                for n in range(row):
                    model.addConstr(var_q_pos[n,p] <= aux_q_pos[n])
                    model.addConstr(-var_q_pos[n,p] <= aux_q_pos[n])
                    model.addConstr(var_q_neg[n,p] <= aux_q_neg[n])
                    model.addConstr(-var_q_neg[n,p] <= aux_q_neg[n])
                    
            model.addConstr(grb.quicksum(aux_q_pos[n] + aux_q_neg[n] + self.kappa[g]*var_b_neg[n] for n in range(row)) <=\
                           row*self.epsilon[g])
        
        for n in range(row):
            model.addConstr(var_b_pos[n] + var_b_neg[n] == 1)
            
            
        for n in range(row):
            for p in range(col):
                model.addConstr(var_b_pos[n]*(x_train[n,p] - 1) - var_q_pos[n,p] <= 0)
                model.addConstr(var_b_pos[n]*x_train[n,p] - var_q_pos[n,p] >= 0)
                model.addConstr(var_b_neg[n]*(x_train[n,p] - 1) - var_q_neg[n,p] <= 0)
                model.addConstr(var_b_neg[n]*x_train[n,p] - var_q_neg[n,p] >= 0)
            
        # Step 4: define objective value
        temp_sum = grb.quicksum(\
            (var_b_pos[n] - var_b_neg[n])*(-y_train[n])*grb.quicksum(self.w_global[p]*x_train[n,p] for p in range(col))\
            + (-y_train[n])*grb.quicksum(self.w_global[p]*(var_q_pos[n,p] - var_q_neg[n,p]) for p in range(col))\
                                for n in range(row))
        obj = (1/row)*temp_sum
        model.setObjective(obj, grb.GRB.MAXIMIZE)
        
        # Step 5: solve the problem
        model.optimize()
        
        # Step 6: store results
        b_pos = np.ones([row])
        b_neg = np.ones([row])
        q_pos = np.ones([row,col])
        q_neg = np.ones([row,col])
        
        for n in range(row):
            b_pos[n] = var_b_pos[n].x
            b_neg[n] = var_b_neg[n].x
            for p in range(col):
                q_pos[n,p] = var_q_pos[n,p].x
                q_neg[n,p] = var_q_neg[n,p].x
                
        temp = {'b_pos': b_pos, 'b_neg': b_neg, 'q_pos': q_pos, 'q_neg': q_neg, 'objective': model.ObjVal, 
                'diagnosis': model.status}
        optimal.update(temp)
        
        model.close()
        return optimal
        
    def subgrad_comp(self,data,opt_dict):
        x_train = data['x']
        y_train = data['y']

        b_pos = opt_dict['b_pos']
        b_neg = opt_dict['b_neg']
        q_pos = opt_dict['q_pos']
        q_neg = opt_dict['q_neg']

        row,col = q_pos.shape

        temp_A = np.zeros([row,col])
        temp_B = np.zeros([row,col])

        for n in range(row):
            z_pos = x_train[n,:] - (1/b_pos[n])*q_pos[n,:]
            z_neg = x_train[n,:] - (1/b_neg[n])*q_neg[n,:]

            if 1 - y_train[n]*np.matmul(self.w_global,z_pos) > 0:
                temp_A[n,:] = -b_pos[n]*y_train[n]*z_pos

            if 1 + y_train[n]*np.matmul(self.w_global,z_neg) > 0:
                temp_B[n,:] = b_neg[n]*y_train[n]*z_neg

        temp_sum_A = np.sum(temp_A,axis=0)
        temp_sum_B = np.sum(temp_B,axis=0)
        subgrad = (1/row)*(temp_sum_A + temp_sum_B)

        return subgrad

    def master_comp(self,all_subgrads):
        agg_subgrad = np.zeros([len(all_subgrads[0,:])])
        for g in range(self.G):
            agg_subgrad = agg_subgrad + self.alphag_vec[g]*all_subgrads[g,:]

        return agg_subgrad
            
    def train_clients(self,train_data):
        sample_x = train_data['x0']
        row_sample,col_sample = sample_x.shape
        all_subgrads = np.zeros([self.G,col_sample])
        runtime_vec = np.zeros(self.G)
        
        for g in range(self.G):
            train_data_g = {'x':train_data['x'+str(g)], 'y':train_data['y'+str(g)]}
            start_g = timeit.default_timer()
            optimal_g = self.extremal_dist(train_data_g,g)
            subgrad_g = self.subgrad_comp(train_data_g,optimal_g)
            stop_g = timeit.default_timer()
            all_subgrads[g,:] = subgrad_g
            runtime_vec[g] = stop_g - start_g

        return all_subgrads, runtime_vec
        
    def train(self,train_data):
        runtime_vec = np.zeros(self.G)
        for t in range(self.max_iter):
            all_subgrads,runtime_vec_t = self.train_clients(train_data)
            runtime_vec = runtime_vec + runtime_vec_t
            agg_subgrad = self.master_comp(all_subgrads)
            self.w_global = self.w_global - (self.stepsize/np.sqrt(t+1))*(agg_subgrad)
            
        adj = np.sum(runtime_vec) - np.max(runtime_vec)
        return self.w_global, adj
        
    def test(self,test_data):
        x_test = test_data['x']
        y_test = test_data['y']
        row_x,col_x = x_test.shape

        y_pred = np.ones(np.shape(y_test))

        for n in range(row_x):
            pred_score = np.sum(self.w_global*x_test[n])
            if pred_score < 0:
                y_pred[n] = -1
                
        acc = sklearn.metrics.f1_score(y_test,y_pred,average='binary')


        return y_pred,acc

    def generate_conf_mat(self,y_true,y_pred):
        """y_true: N*1 array of true labels
           y_pred: N*1 array of predicted labels"""

        conf_mat = sklearn.metrics.confusion_matrix(y_true,y_pred)
        disp_conf_mat = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=conf_mat)

        return conf_mat,disp_conf_mat
    

class ADMM_alg:
    """ADMM-Based Optimization Algorithm"""
    
    def __init__(self,param,init_vars,Ng_vec):
        self.epsilon = param['epsilon']
        self.kappa = param['kappa']
        self.pnorm = param['pnorm']
        self.rho = param['rho']
        self.max_iter = param['max_iter']
        self.tau = param['tau']
        self.w_global = init_vars['w']
        self.mu = init_vars['mu']
        self.G = len(Ng_vec)
        self.N = np.sum(Ng_vec)
        self.alphag_vec = np.ones(np.shape(Ng_vec))*(1/self.G)
        self.Ng_vec = Ng_vec
        
    def train_clients(self,train_data):
        sample_x = train_data['x0']
        row_sample,col_sample = sample_x.shape
        self.w_local = np.zeros([self.G,col_sample])
        runtime_vec = np.zeros(self.G)
        
        for g in range(self.G):
            train_data_g = {'x':train_data['x'+str(g)], 'y':train_data['y'+str(g)]}
            start_g = timeit.default_timer()
            optimal_g = self.client_DR_SVM(train_data_g,g)
            stop_g = timeit.default_timer()
            self.w_local[g,:] = optimal_g['w']
            runtime_vec[g] = stop_g - start_g
            
        return runtime_vec

        
    def client_DR_SVM(self,data,g):
        """Distributionally robust binary SVM without support information"""
        
        x_train = data['x']
        y_train = data['y'].flatten()

        row, col = x_train.shape
        optimal = {}
        

        # Step 0: create model
        model = grb.Model('DRSVM_without_support_'+str(g))
        model.setParam('OutputFlag', False)

        # Step 1: define decision variables
        var_lambda = model.addVar(vtype=grb.GRB.CONTINUOUS)
        var_s = {}
        var_w = {}
        slack_var = {}
        for n in range(row):
            var_s[n] = model.addVar(vtype=grb.GRB.CONTINUOUS,lb=0)
        for p in range(col):
            var_w[p] = model.addVar(
                vtype=grb.GRB.CONTINUOUS, lb=-grb.GRB.INFINITY)
            if self.pnorm == 1:
                slack_var[p] = model.addVar(vtype=grb.GRB.CONTINUOUS)

        # Step 2: integerate variables
        model.update()

        # Step 3: define constraints
        for n in range(row):
            model.addConstr(
                1 - y_train[n] * grb.quicksum(var_w[p] * x_train[n,p]
                                              for p in range(col)) <= var_s[n])
            model.addConstr(
                1 + y_train[n] * grb.quicksum(var_w[p] * x_train[n, p]
                                              for p in range(col)) -
                self.kappa[g] * var_lambda <= var_s[n])

        if self.pnorm == 1:
            for p in range(col):
                model.addConstr(var_w[p] <= slack_var[p])
                model.addConstr(-var_w[p] <= slack_var[p])
            model.addConstr(grb.quicksum(slack_var[p]
                                         for p in range(col)) <= var_lambda)
        elif self.pnorm == 2:
            model.addQConstr(
                grb.quicksum(var_w[p] * var_w[p]
                             for p in range(col)) <= var_lambda * var_lambda)

        elif self.pnorm == float('Inf'):
            for p in range(col):
                model.addConstr(var_w[p] <= var_lambda)
                model.addConstr(-var_w[p] <= var_lambda)

        # Step 4: define objective value
        sum_var_s = grb.quicksum(var_s[n] for n in range(row))
        norm_regularizer = grb.quicksum((var_w[p] - self.w_global[p] + self.mu[g][p])*\
                                       (var_w[p] - self.w_global[p] + self.mu[g][p]) for p in range(col))
        obj = var_lambda*self.epsilon[g] + (1/row)*sum_var_s + (self.rho/2)*norm_regularizer + self.tau*grb.quicksum(var_w[p]*var_w[p] for p in range(col))
        model.setObjective(obj, grb.GRB.MINIMIZE)

        # Step 5: solve the problem
        model.optimize()

        # Step 6: store results
        w_opt = np.array([var_w[p].x for p in range(col)])
        tmp = {'w': w_opt,'objective': model.ObjVal,'diagnosis': model.status}
        optimal.update(tmp)
        
        model.close()
        return optimal
    
    def master_comp(self):
        temp_w_global = np.zeros(np.shape(self.w_global))
        for g in range(self.G):
            temp_w_global = temp_w_global + self.alphag_vec[g]*(self.w_local[g]+self.mu[g])
        
        self.w_global = temp_w_global
        
    def update_mu(self):
        runtime_vec = np.zeros(self.G)
        for g in range(self.G):
            start_g = timeit.default_timer()
            self.mu[g,:] = self.mu[g,:] + self.w_local[g,:] - self.w_global
            stop_g = timeit.default_timer()
            runtime_vec[g] = stop_g - start_g
            
        return runtime_vec
            
            
    def train(self,train_data):
        runtime_vec = np.zeros(self.G)
        for t in range(self.max_iter):
            runtime_vec_A_t = self.train_clients(train_data)
            runtime_vec = runtime_vec + runtime_vec_A_t
            self.master_comp()
            runtime_vec_B_t = self.update_mu()
            runtime_vec = runtime_vec + runtime_vec_B_t
            
        adj = np.sum(runtime_vec) - np.max(runtime_vec)
        return self.w_global, adj

    
    def test(self,test_data):
        x_test = test_data['x']
        y_test = test_data['y']
        row_x,col_x = x_test.shape

        y_pred = np.ones(np.shape(y_test))

        for n in range(row_x):
            pred_score = np.sum(self.w_global*x_test[n])
            if pred_score < 0:
                y_pred[n] = -1
                
        acc = sklearn.metrics.f1_score(y_test,y_pred,average='binary')


        return y_pred,acc
    
    def generate_conf_mat(self,y_true,y_pred):
        """y_true: N*1 array of true labels
           y_pred: N*1 array of predicted labels"""
            
        conf_mat = sklearn.metrics.confusion_matrix(y_true,y_pred)
        disp_conf_mat = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=conf_mat)
        
        return conf_mat,disp_conf_mat
    

class ADMM_alg_reg:
    """ADMM-Based Optimization Algorithm"""
    
    def __init__(self,param,init_vars,Ng_vec):
        self.epsilon = param['epsilon']
        self.kappa = param['kappa']
        self.pnorm = param['pnorm']
        self.rho = param['rho']
        self.max_iter = param['max_iter']
        self.w_global = init_vars['w']
        self.mu = init_vars['mu']
        self.G = len(Ng_vec)
        self.N = np.sum(Ng_vec)
        self.alphag_vec = np.ones(np.shape(Ng_vec))*(1/self.G)
        self.Ng_vec = Ng_vec
        
    def train_clients(self,train_data):
        sample_x = train_data['x0']
        row_sample,col_sample = sample_x.shape
        self.w_local = np.zeros([self.G,col_sample])
        runtime_vec = np.zeros(self.G)
        
        for g in range(self.G):
            train_data_g = {'x':train_data['x'+str(g)], 'y':train_data['y'+str(g)]}
            start_g = timeit.default_timer()
            optimal_g = self.client_DR_SVM(train_data_g,g)
            stop_g = timeit.default_timer()
            self.w_local[g,:] = optimal_g['w']
            runtime_vec[g] = stop_g - start_g
            
        return runtime_vec

        
    def client_DR_SVM(self,data,g):
        """Distributionally robust binary SVM without support information"""
        
        x_train = data['x']
        y_train = data['y'].flatten()

        row, col = x_train.shape
        optimal = {}
        

        # Step 0: create model
        model = grb.Model('DRSVM_without_support_'+str(g))
        model.setParam('OutputFlag', False)

        # Step 1: define decision variables
        var_lambda = model.addVar(vtype=grb.GRB.CONTINUOUS)
        var_s = {}
        var_w = {}
        slack_var = {}
        for n in range(row):
            var_s[n] = model.addVar(vtype=grb.GRB.CONTINUOUS,lb=0)
        for p in range(col):
            var_w[p] = model.addVar(
                vtype=grb.GRB.CONTINUOUS, lb=-grb.GRB.INFINITY)
            if self.pnorm == 1:
                slack_var[p] = model.addVar(vtype=grb.GRB.CONTINUOUS)

        # Step 2: integerate variables
        model.update()

        # Step 3: define constraints
        for n in range(row):
            model.addConstr(
                1 - y_train[n] * grb.quicksum(var_w[p] * x_train[n,p]
                                              for p in range(col)) <= var_s[n])

        if self.pnorm == 1:
            for p in range(col):
                model.addConstr(var_w[p] <= slack_var[p])
                model.addConstr(-var_w[p] <= slack_var[p])
            model.addConstr(grb.quicksum(slack_var[p]
                                         for p in range(col)) <= var_lambda)
        elif self.pnorm == 2:
            model.addQConstr(
                grb.quicksum(var_w[p] * var_w[p]
                             for p in range(col)) <= var_lambda * var_lambda)

        elif self.pnorm == float('Inf'):
            for p in range(col):
                model.addConstr(var_w[p] <= var_lambda)
                model.addConstr(-var_w[p] <= var_lambda)

        # Step 4: define objective value
        sum_var_s = grb.quicksum(var_s[n] for n in range(row))
        norm_regularizer = grb.quicksum((var_w[p] - self.w_global[p] + self.mu[g][p])*\
                                       (var_w[p] - self.w_global[p] + self.mu[g][p]) for p in range(col))
        obj = var_lambda*self.epsilon[g] + (1/row)*sum_var_s + (self.rho/2)*norm_regularizer
        model.setObjective(obj, grb.GRB.MINIMIZE)

        # Step 5: solve the problem
        model.optimize()

        # Step 6: store results
        w_opt = np.array([var_w[p].x for p in range(col)])
        tmp = {'w': w_opt,'objective': model.ObjVal,'diagnosis': model.status}
        optimal.update(tmp)
        
        model.close()
        return optimal
    
    def master_comp(self):
        temp_w_global = np.zeros(np.shape(self.w_global))
        for g in range(self.G):
            temp_w_global = temp_w_global + self.alphag_vec[g]*(self.w_local[g]+self.mu[g])
        
        self.w_global = temp_w_global
        
    def update_mu(self):
        runtime_vec = np.zeros(self.G)
        for g in range(self.G):
            start_g = timeit.default_timer()
            self.mu[g,:] = self.mu[g,:] + self.w_local[g,:] - self.w_global
            stop_g = timeit.default_timer()
            runtime_vec[g] = stop_g - start_g
            
        return runtime_vec
            
            
    def train(self,train_data):
        runtime_vec = np.zeros(self.G)
        for t in range(self.max_iter):
            runtime_vec_A_t = self.train_clients(train_data)
            runtime_vec = runtime_vec + runtime_vec_A_t
            self.master_comp()
            runtime_vec_B_t = self.update_mu()
            runtime_vec = runtime_vec + runtime_vec_B_t
            
        adj = np.sum(runtime_vec) - np.max(runtime_vec)
        return self.w_global, adj

    
    def test(self,test_data):
        x_test = test_data['x']
        y_test = test_data['y']
        row_x,col_x = x_test.shape
        
        y_pred = np.ones(np.shape(y_test))
        
        for n in range(row_x):
            pred_score = np.sum(self.w_global*x_test[n])
            if pred_score < 0:
                y_pred[n] = -1
                
                
        acc = sklearn.metrics.f1_score(y_test,y_pred,average='binary')
        
        return y_pred,acc
    
    def generate_conf_mat(self,y_true,y_pred):
        """y_true: N*1 array of true labels
           y_pred: N*1 array of predicted labels"""
            
        conf_mat = sklearn.metrics.confusion_matrix(y_true,y_pred)
        disp_conf_mat = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=conf_mat)
        
        return conf_mat,disp_conf_mat
    
    
class fedAvg:
    def __init__(self,param,init_vars,Ng_vec):
        self.epsilon = param['epsilon']
        self.learning_rate = param['lr']
        self.batch_size = param['batch_size']
        self.local_steps = param['local_steps']
        self.max_iter = param['max_iter']
        self.w_global = init_vars['w']
        self.G = len(Ng_vec)
        self.N = np.sum(Ng_vec)
        self.alphag_vec = np.ones(np.shape(Ng_vec))*(1/self.G)
        self.Ng_vec = Ng_vec
        
    def train_clients(self,train_data):
        sample_x = train_data['x0']
        row_sample,col_sample = sample_x.shape
        self.w_local = np.zeros([self.G,col_sample])
        self.grad_local = np.zeros([self.G,col_sample])
        
        for g in range(self.G):
            train_data_g = {'x':train_data['x'+str(g)], 'y':train_data['y'+str(g)]}
            w_local_g, grad_local_g = self.client_train(train_data_g,g)
            self.w_local[g,:] = w_local_g
            self.grad_local[g,:] = grad_local_g
            
    def client_train(self,data,g):
        x_train = data['x']
        y_train = data['y']
        
        w_curr = np.copy(self.w_global)
        
        
        for s in range(self.local_steps):
            all_idx = np.arange(len(x_train))
            for i in range(int(1/self.batch_size)):
                batch_idx = np.random.choice(all_idx, int(len(x_train)*self.batch_size), replace=False)
                x_batch = x_train[batch_idx]
                y_batch = y_train[batch_idx]
                
                grad_curr = self.comp_subgrad(w_curr,x_batch,y_batch,g)
                w_new = w_curr - (self.learning_rate/np.sqrt(s+1))*grad_curr
                w_curr = np.copy(w_new)
                all_idx = np.setdiff1d(all_idx, batch_idx, assume_unique=True)
            
        grad_curr = self.comp_subgrad(w_curr,x_batch,y_batch,g)
        
        return w_curr, grad_curr
    
    def comp_subgrad(self,w_curr,x_batch,y_batch,g):
         
        subgrad_temp = np.zeros(w_curr.shape)
        
        for n in range(len(x_batch)):
            if 1 - y_batch[n]*np.matmul(w_curr,x_batch[n]) > 0:
                subgrad_temp = subgrad_temp - y_batch[n]*x_batch[n]
                
        subgrad_final = (1/len(x_batch))*subgrad_temp + 2*self.epsilon[g]*w_curr
        
        return subgrad_final
    
    def master_comp(self):
        temp_w_global = np.zeros(np.shape(self.w_global))
        for g in range(self.G):
            temp_w_global = temp_w_global + self.alphag_vec[g]*(self.w_local[g])
            
        self.w_global = temp_w_global
        
    def train(self,train_data):
        for t in range(self.max_iter):
            self.train_clients(train_data)
            self.master_comp()
            
        return self.w_global
    
    def test(self,test_data):
        x_test = test_data['x']
        y_test = test_data['y']
        row_x,col_x = x_test.shape
        
        y_pred = np.ones(np.shape(y_test))
        
        for n in range(row_x):
            pred_score = np.sum(self.w_global*x_test[n])
            if pred_score < 0:
                y_pred[n] = -1
                
                
        acc = sklearn.metrics.f1_score(y_test,y_pred,average='binary')
        
        return y_pred,acc
    
    def generate_conf_mat(self,y_true,y_pred):
        """y_true: N*1 array of true labels
           y_pred: N*1 array of predicted labels"""
            
        conf_mat = sklearn.metrics.confusion_matrix(y_true,y_pred)
        disp_conf_mat = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=conf_mat)
        
        return conf_mat,disp_conf_mat
    
class fedSGD:
    def __init__(self,param,init_vars,Ng_vec):
        self.epsilon = param['epsilon']
        self.learning_rate = param['lr']
        self.batch_size = param['batch_size']
        self.local_steps = param['local_steps']
        self.max_iter = param['max_iter']
        self.w_global = init_vars['w']
        self.G = len(Ng_vec)
        self.N = np.sum(Ng_vec)
        self.alphag_vec = np.ones(np.shape(Ng_vec))*(1/self.G)
        self.Ng_vec = Ng_vec
        
    def train_clients(self,train_data):
        sample_x = train_data['x0']
        row_sample,col_sample = sample_x.shape
        self.w_local = np.zeros([self.G,col_sample])
        self.grad_local = np.zeros([self.G,col_sample])
        runtime_vec = np.zeros(self.G)
        
        for g in range(self.G):
            train_data_g = {'x':train_data['x'+str(g)], 'y':train_data['y'+str(g)]}
            w_local_g, grad_local_g = self.client_train(train_data_g,g)
            self.w_local[g,:] = w_local_g
            self.grad_local[g,:] = grad_local_g
            
    def client_train(self,data,g):
        x_train = data['x']
        y_train = data['y']
        
        w_curr = np.copy(self.w_global)
        
        x_batch = x_train[:]
        y_batch = y_train[:]
        
        grad_curr = self.comp_subgrad(w_curr,x_batch,y_batch,g)
        
        return w_curr, grad_curr
    
    def comp_subgrad(self,w_curr,x_batch,y_batch,g):
         
        subgrad_temp = np.zeros(w_curr.shape)
        
        for n in range(len(x_batch)):
            if 1 - y_batch[n]*np.matmul(w_curr,x_batch[n]) > 0:
                subgrad_temp += -y_batch[n]*x_batch[n]
                
        subgrad_final = (1/len(x_batch))*subgrad_temp + 2*self.epsilon[g]*w_curr
        
        return subgrad_final
    
    def master_comp(self,t):
        temp_grad_global = np.zeros(np.shape(self.w_global))
        for g in range(self.G):
            temp_grad_global = temp_grad_global + self.alphag_vec[g]*(self.grad_local[g])
            
        self.w_global = self.w_global - (self.learning_rate/np.sqrt(t+1))*temp_grad_global
        
    def train(self,train_data):
        for t in range(self.max_iter):
            self.train_clients(train_data)
            self.master_comp(t)
            
        return self.w_global
    
    def test(self,test_data):
        x_test = test_data['x']
        y_test = test_data['y']
        row_x,col_x = x_test.shape
        
        y_pred = np.ones(np.shape(y_test))
        
        for n in range(row_x):
            pred_score = np.sum(self.w_global*x_test[n])
            if pred_score < 0:
                y_pred[n] = -1
                
                
        acc = sklearn.metrics.f1_score(y_test,y_pred,average='binary')
        
        return y_pred,acc
    
    def generate_conf_mat(self,y_true,y_pred):
        """y_true: N*1 array of true labels
           y_pred: N*1 array of predicted labels"""
            
        conf_mat = sklearn.metrics.confusion_matrix(y_true,y_pred)
        disp_conf_mat = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=conf_mat)
        
        return conf_mat,disp_conf_mat
    
        
class fedProx:
    def __init__(self,param,init_vars,Ng_vec):
        self.epsilon = param['epsilon']
        self.learning_rate = param['lr']
        self.batch_size = param['batch_size']
        self.local_steps = param['local_steps']
        self.max_iter = param['max_iter']
        self.mu = param['mu']
        self.w_global = init_vars['w']
        self.G = len(Ng_vec)
        self.N = np.sum(Ng_vec)
        self.alphag_vec = np.ones(np.shape(Ng_vec))*(1/self.G)
        self.Ng_vec = Ng_vec
        
    def train_clients(self,train_data):
        sample_x = train_data['x0']
        row_sample,col_sample = sample_x.shape
        self.w_local = np.zeros([self.G,col_sample])
        self.grad_local = np.zeros([self.G,col_sample])
        runtime_vec = np.zeros(self.G)
        
        for g in range(self.G):
            train_data_g = {'x':train_data['x'+str(g)], 'y':train_data['y'+str(g)]}
            w_local_g, grad_local_g = self.client_train(train_data_g,g)
            self.w_local[g,:] = w_local_g
            self.grad_local[g,:] = grad_local_g
            
    def client_train(self,data,g):
        x_train = data['x']
        y_train = data['y']
        
        w_curr = np.copy(self.w_global)
        
        for s in range(self.local_steps):
            all_idx = np.arange(len(x_train))
            for i in range(int(1/self.batch_size)):
                batch_idx = np.random.choice(all_idx, int(len(x_train)*self.batch_size), replace=False)
                x_batch = x_train[batch_idx]
                y_batch = y_train[batch_idx]
                
                grad_curr = self.comp_subgrad(w_curr,x_batch,y_batch,g)
                w_new = w_curr - (self.learning_rate/np.sqrt(s+1))*grad_curr
                w_curr = np.copy(w_new)
                all_idx = np.setdiff1d(all_idx, batch_idx, assume_unique=True)
            
        grad_curr = self.comp_subgrad(w_curr,x_batch,y_batch,g)
        
        return w_curr, grad_curr
    
    def comp_subgrad(self,w_curr,x_batch,y_batch,g):
        subgrad_temp = np.zeros(w_curr.shape)

        for n in range(len(x_batch)):
            if 1 - y_batch[n]*np.matmul(w_curr,x_batch[n]) > 0:
                subgrad_temp += -y_batch[n]*x_batch[n]

        subgrad_final = (1/len(x_batch))*subgrad_temp + 2*self.epsilon[g]*w_curr + self.mu*(w_curr - self.w_global)

        return subgrad_final
    
    def master_comp(self):
        temp_w_global = np.zeros(np.shape(self.w_global))
        for g in range(self.G):
            temp_w_global = temp_w_global + self.alphag_vec[g]*(self.w_local[g])
            
        self.w_global = temp_w_global
        
    def train(self,train_data):
        for t in range(self.max_iter):
            self.train_clients(train_data)
            self.master_comp()
            
        return self.w_global
    
    def test(self,test_data):
        x_test = test_data['x']
        y_test = test_data['y']
        row_x,col_x = x_test.shape
        
        y_pred = np.ones(np.shape(y_test))
        
        for n in range(row_x):
            pred_score = np.sum(self.w_global*x_test[n])
            if pred_score < 0:
                y_pred[n] = -1
                
                
        acc = sklearn.metrics.f1_score(y_test,y_pred,average='binary')
        
        return y_pred,acc
    
    def generate_conf_mat(self,y_true,y_pred):
        """y_true: N*1 array of true labels
           y_pred: N*1 array of predicted labels"""
            
        conf_mat = sklearn.metrics.confusion_matrix(y_true,y_pred)
        disp_conf_mat = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=conf_mat)
        
        return conf_mat,disp_conf_mat
    

class fedDRO_kl:
    def __init__(self,param,init_vars,Ng_vec):
        self.learning_rate = param['lr']
        self.batch_size = param['batch_size']
        self.local_steps = param['local_steps']
        self.max_iter = param['max_iter']
        self.lam = param['lam']
        self.beta = param['beta']
        self.w_global = init_vars['w']
        self.z_init = init_vars['z']
        self.G = len(Ng_vec)
        self.N = np.sum(Ng_vec)
        self.alphag_vec = np.ones(np.shape(Ng_vec))*(1/self.G)
        self.Ng_vec = Ng_vec
        
    def comp_z(self,t,z_old,train_data,w_curr):
        num_samp = len(train_data['x'])
        tot_loss = 0
        
        for i in range(num_samp):
            curr_loss = 1 - train_data['y'][i] * np.sum(train_data['x'][i]*w_curr)
            if curr_loss > 0:
                g_func = math.exp(curr_loss/self.lam)
                tot_loss += g_func
                
        emp_exp = (1/num_samp)*tot_loss
        
        z_new = (1 - self.beta)*(z_old - emp_exp) + emp_exp
        
        return z_new
    
    def comp_grad(self,z_agg,train_data,w_curr):
        num_samp = len(train_data['x'])
        grad_f = (1/z_agg)
        tot_grad = 0
        
        for i in range(num_samp):
            curr_loss = 1 - train_data['y'][i] * np.sum(train_data['x'][i]*w_curr)
            if curr_loss > 0:
                grad_inner = -train_data['y'][i]*train_data['x'][i]
                grad_y = math.exp(curr_loss/self.lam)*(1/self.lam)*grad_inner
                curr_grad = grad_y*grad_f
                
                tot_grad += curr_grad
                
        final_grad = (1/num_samp)*tot_grad
        
        return final_grad
    
    def train(self,train_data):
        sample_x = train_data['x0']
        row_sample,col_sample = sample_x.shape
        z_curr = np.copy(self.z_init)
        self.w_local = np.zeros([self.G,col_sample])
        
        for t in range(self.max_iter*self.local_steps):
            batch_idx_dict = {}
            z_new = np.zeros(z_curr.shape)
            for g in range(self.G):
                str_x = 'x'+str(g)
                str_y = 'y'+str(g)
                x_local = train_data[str_x]
                y_local = train_data[str_y]

                batch_samp_num = int(self.batch_size*len(x_local))
                batch_idx = np.random.choice(len(x_local), batch_samp_num, replace=False)
                batch_idx_dict['client_'+str(g)] = batch_idx
                x_batch = x_local[batch_idx]
                y_batch = y_local[batch_idx]
                local_data = {'x': x_batch, 'y':y_batch}
                z_g_new = self.comp_z(t,z_curr,local_data,self.w_local[g])
                
                z_new += z_g_new
            
            z_new = (1/self.G)*z_new
            z_curr = np.copy(z_new)
            
            for g in range(self.G):
                str_x = 'x'+str(g)
                str_y = 'y'+str(g)
                x_local = train_data[str_x]
                y_local = train_data[str_y]
                batch_idx = batch_idx_dict['client_'+str(g)]
                x_batch = x_local[batch_idx]
                y_batch = y_local[batch_idx]
                local_data = {'x': x_batch, 'y':y_batch}
                local_grad = self.comp_grad(z_curr,local_data,self.w_local[g])
                
                
                self.w_local[g] = self.w_local[g] - (self.learning_rate/np.sqrt(t+1))*local_grad
                
            if (t+1)%self.local_steps == 0:
                self.w_global = np.mean(self.w_local,axis=0)
                for g in range(self.G):
                    self.w_local[g] = self.w_global
                    
        self.w_global = np.mean(self.w_local,axis=0)
        
        return self.w_global
    
    
    def test(self,test_data):
        x_test = test_data['x']
        y_test = test_data['y']
        row_x,col_x = x_test.shape
        
        y_pred = np.ones(np.shape(y_test))
        
        for n in range(row_x):
            pred_score = np.sum(self.w_global*x_test[n])
            if pred_score < 0:
                y_pred[n] = -1
                
                
        acc = sklearn.metrics.f1_score(y_test,y_pred,average='binary')
        
        return y_pred,acc
    
    def generate_conf_mat(self,y_true,y_pred):
        """y_true: N*1 array of true labels
           y_pred: N*1 array of predicted labels"""
            
        conf_mat = sklearn.metrics.confusion_matrix(y_true,y_pred)
        disp_conf_mat = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=conf_mat)
        
        return conf_mat,disp_conf_mat
    
class centralized_classifier:
    
    def __init__(self,param):
        self.epsilon = param['epsilon']
        self.kappa = param['kappa']
        self.pnorm = param['pnorm']
        
    def train(self,data):
        x_train = data['x']
        y_train = data['y'].flatten()

        row, col = x_train.shape
        optimal = {}


        # Step 0: create model
        model = grb.Model('DRSVM_without_support_')
        model.setParam('OutputFlag', False)

        # Step 1: define decision variables
        var_lambda = model.addVar(vtype=grb.GRB.CONTINUOUS)
        var_s = {}
        var_w = {}
        slack_var = {}
        for n in range(row):
            var_s[n] = model.addVar(vtype=grb.GRB.CONTINUOUS,lb=0)
        for p in range(col):
            var_w[p] = model.addVar(
                vtype=grb.GRB.CONTINUOUS, lb=-grb.GRB.INFINITY)
            if self.pnorm == 1:
                slack_var[p] = model.addVar(vtype=grb.GRB.CONTINUOUS)

        # Step 2: integerate variables
        model.update()

        # Step 3: define constraints
        for n in range(row):
            model.addConstr(
                1 - y_train[n] * grb.quicksum(var_w[p] * x_train[n,p]
                                              for p in range(col)) <= var_s[n])
            model.addConstr(
                1 + y_train[n] * grb.quicksum(var_w[p] * x_train[n, p]
                                              for p in range(col)) -
                self.kappa * var_lambda <= var_s[n])

        if self.pnorm == 1:
            for p in range(col):
                model.addConstr(var_w[p] <= slack_var[p])
                model.addConstr(-var_w[p] <= slack_var[p])
            model.addConstr(grb.quicksum(slack_var[p]
                                         for p in range(col)) <= var_lambda)
        elif self.pnorm == 2:
            model.addQConstr(
                grb.quicksum(var_w[p] * var_w[p]
                             for p in range(col)) <= var_lambda * var_lambda)

        elif self.pnorm == float('Inf'):
            for p in range(col):
                model.addConstr(var_w[p] <= var_lambda)
                model.addConstr(-var_w[p] <= var_lambda)

        # Step 4: define objective value
        sum_var_s = grb.quicksum(var_s[n] for n in range(row))
        obj = var_lambda*self.epsilon + (1/row)*sum_var_s
        model.setObjective(obj, grb.GRB.MINIMIZE)

        # Step 5: solve the problem
        model.optimize()

        # Step 6: store results
        w_opt = np.array([var_w[p].x for p in range(col)])
        tmp = {'w': w_opt,'objective': model.ObjVal,'diagnosis': model.status}
        optimal.update(tmp)
        self.w_opt = w_opt

        model.close()
        return self.w_opt

    def test(self,test_data):
        x_test = test_data['x']
        y_test = test_data['y']
        row_x,col_x = x_test.shape
        
        y_pred = np.ones(np.shape(y_test))
        
        for n in range(row_x):
            pred_score = np.sum(self.w_opt*x_test[n])
            if pred_score < 0:
                y_pred[n] = -1
                
        acc = sklearn.metrics.f1_score(y_test,y_pred,average='binary')
        
        return y_pred,acc
    
    def generate_conf_mat(self,y_true,y_pred):
        """y_true: N*1 array of true labels
           y_pred: N*1 array of predicted labels"""
            
        conf_mat = sklearn.metrics.confusion_matrix(y_true,y_pred)
        disp_conf_mat = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=conf_mat)
        
        return conf_mat,disp_conf_mat
    

    
    
def save_exp_file(P,G,perc_train,perc_test,perc_g_vec,num_exp,max_iter_vec,
                 param_vec,acc_vec_fedAvg,acc_vec_fedSGD,acc_vec_fedProx,
                 acc_vec_fedDRO_kl,acc_vec_SM,acc_vec_ADMM,acc_vec_ADMM_sc,
                 acc_vec_central,batch_size,local_steps,mu):
    exp_dict = {
        'P':P,
        'G':G,
        'N_train':perc_train,
        'N_test':perc_test,
        'perc_g_vec':perc_g_vec,
        'num_exp':num_exp,
        'max_iter_vec':max_iter_vec,
        'param_vec':param_vec,
        'acc_vec_fedAvg':acc_vec_fedAvg,
        'acc_vec_fedSGD':acc_vec_fedSGD,
        'acc_vec_fedProx':acc_vec_fedProx,
        'acc_vec_fedDRO_kl':acc_vec_fedDRO_kl,
        'acc_vec_SM':acc_vec_SM,
        'acc_vec_ADMM':acc_vec_ADMM,
        'acc_vec_ADMM_sc':acc_vec_ADMM_sc,
        'acc_vec_central':acc_vec_central,
        'batch_size':batch_size,
        'local_steps':local_steps,
        'mu':mu
    }

    filename = 'FDR-SVM_Banknote_'+str(int(timeit.default_timer()))+str(np.random.randint(0,high=1000))+'.mat'
    scipy.io.savemat(filename,exp_dict)
    
def cv_split(client_sets,client_sets_norm,G=4,n_folds=5):
    client_sets_split = {}
    client_sets_norm_split = {}
    
    client_test_split = {}
    client_test_norm_split = {}
    
    val_sets = {}
    val_sets_norm = {}
    
    for g in range(G):
        idx = np.random.permutation(len(client_sets['x'+str(g)]))
        x_g = client_sets['x'+str(g)][idx]
        y_g = client_sets['y'+str(g)][idx]
        
        x_g_norm = client_sets_norm['x'+str(g)][idx]
        y_g_norm = client_sets_norm['y'+str(g)][idx]
        kf = sklearn.model_selection.KFold(n_splits=n_folds, shuffle=False)
        
        ct = 0
        for train_index, val_index in kf.split(x_g):
            client_sets_split['x'+str(g)+'_'+str(ct)], client_test_split['x'+str(g)+'_'+str(ct)] = x_g[train_index], x_g[val_index]
            client_sets_norm_split['x'+str(g)+'_'+str(ct)], client_test_norm_split['x'+str(g)+'_'+str(ct)] = x_g_norm[train_index], x_g_norm[val_index]
            
            client_sets_split['y'+str(g)+'_'+str(ct)], client_test_split['y'+str(g)+'_'+str(ct)] = y_g[train_index], y_g[val_index]
            client_sets_norm_split['y'+str(g)+'_'+str(ct)], client_test_norm_split['y'+str(g)+'_'+str(ct)] = y_g_norm[train_index], y_g_norm[val_index]
            
            ct += 1
            
    for i in range(n_folds):
        x_val_i = np.concatenate([client_test_split['x'+str(0)+'_'+str(i)],client_test_split['x'+str(1)+'_'+str(i)],
                                 client_test_split['x'+str(2)+'_'+str(i)],client_test_split['x'+str(3)+'_'+str(i)]], axis=0)
        
        y_val_i = np.concatenate([client_test_split['y'+str(0)+'_'+str(i)],client_test_split['y'+str(1)+'_'+str(i)],
                                 client_test_split['y'+str(2)+'_'+str(i)],client_test_split['y'+str(3)+'_'+str(i)]], axis=0)
        
        x_val_norm_i = np.concatenate([client_test_norm_split['x'+str(0)+'_'+str(i)],client_test_norm_split['x'+str(1)+'_'+str(i)],
                                 client_test_norm_split['x'+str(2)+'_'+str(i)],client_test_norm_split['x'+str(3)+'_'+str(i)]], axis=0)
        
        y_val_norm_i = np.concatenate([client_test_norm_split['y'+str(0)+'_'+str(i)],client_test_norm_split['y'+str(1)+'_'+str(i)],
                                 client_test_norm_split['y'+str(2)+'_'+str(i)],client_test_norm_split['y'+str(3)+'_'+str(i)]], axis=0)
        
        val_sets['x_split'+str(i)] = x_val_i
        val_sets['y_split'+str(i)] = y_val_i
        
        val_sets_norm['x_split'+str(i)] = x_val_norm_i
        val_sets_norm['y_split'+str(i)] = y_val_norm_i
        
    return client_sets_split, client_sets_norm_split, val_sets, val_sets_norm


def cv_split_central(central_set,n_folds):
    central_set_split = {}
    central_val_split = {}
    
    idx = np.random.permutation(len(central_set['x']))
    
    x = central_set['x'][idx]
    y = central_set['y'][idx]
    kf = sklearn.model_selection.KFold(n_splits=n_folds, shuffle=False)
    
    ct = 0
    for train_index, val_index in kf.split(x):
        central_set_split['x_split'+str(ct)], central_val_split['x_split'+str(ct)] = x[train_index], x[val_index]
        central_set_split['y_split'+str(ct)], central_val_split['y_split'+str(ct)] = y[train_index], y[val_index]
        ct += 1
        
    return central_set_split, central_val_split
     
    
        
        
            
G = 4
P = 4
batch_size = 0.2
local_steps = 5
mu = 1
n_folds = 5

perc_train = 0.7
perc_test = 0.3
perc_g_vec = list(np.ones(G)*(1/G))

num_exp = 1
max_iter_vec = [5,10,20,60,100,140,180,220]
max_iter_vec_subgrad = [100,140,180,220]
param_vec = np.logspace(-3,0,4)
param_vec_subgrad = np.logspace(0,3,4)

epsilon_vec = np.logspace(-5,-1,5)
kappa_vec = np.asarray([0.1,0.25,0.5,0.75,1])

init_fed = np.random.rand(P)

acc_vec_fedAvg = np.zeros([num_exp])
acc_vec_fedSGD = np.zeros([num_exp])
acc_vec_fedProx = np.zeros([num_exp])
acc_vec_fedDRO_kl = np.zeros([num_exp])
acc_vec_SM = np.zeros([num_exp])
acc_vec_ADMM = np.zeros([num_exp])
acc_vec_ADMM_sc = np.zeros([num_exp])
acc_vec_central = np.zeros([num_exp])

for i in range(num_exp):
    
    central_set, client_sets, test_set, central_set_norm, client_sets_norm, test_set_norm, Ng_vec = gen_data(G=G,
    perc_train=perc_train,perc_test=perc_test,perc_wrong_y=0,perc_g_vec=perc_g_vec)
    
    epsilon_vec_algs = np.zeros(G)
    for j in range(len(Ng_vec)):
        epsilon_vec_algs[j] = 1/(10*Ng_vec[j])
        
    
    client_sets_split, client_sets_norm_split, val_sets, val_sets_norm = cv_split(client_sets,client_sets_norm,4,5)
    
    central_set_split, central_val_split = cv_split_central(central_set,5)
    
    acc_tensor_fedAvg = np.zeros([len(max_iter_vec),len(param_vec),n_folds])
    acc_tensor_fedSGD = np.zeros([len(max_iter_vec),len(param_vec),n_folds])
    acc_tensor_fedProx = np.zeros([len(max_iter_vec),len(param_vec),n_folds])
    acc_tensor_fedDRO_kl = np.zeros([len(max_iter_vec),len(param_vec),n_folds])
    acc_tensor_SM = np.zeros([len(max_iter_vec),len(param_vec),n_folds])
    acc_tensor_ADMM = np.zeros([len(max_iter_vec),len(param_vec),n_folds])
    acc_tensor_ADMM_sc = np.zeros([len(max_iter_vec),len(param_vec),n_folds])
    
    acc_tensor_central = np.zeros([len(epsilon_vec),len(kappa_vec),n_folds])
    
    for eps in range(len(epsilon_vec)):
        for kap in range(len(kappa_vec)):
            for fol in range(n_folds):
                params_central = {'epsilon':epsilon_vec[eps], 'kappa':kappa_vec[kap], 'pnorm':1}
                
                central_set_curr = {'x': central_set_split['x_split'+str(fol)],
                                    'y': central_set_split['y_split'+str(fol)]}
                
                val_set_curr = {'x': central_val_split['x_split'+str(fol)],
                                'y': central_val_split['y_split'+str(fol)]}
                
                classifier_central = centralized_classifier(params_central)
                w_opt_central = classifier_central.train(central_set_curr)
                y_pred_central, acc_central = classifier_central.test(val_set_curr)
                acc_tensor_central[eps,kap,fol] = acc_central
                
    acc_mat_central = np.mean(acc_tensor_central,axis=2)
    
    eps_central, kap_central = np.unravel_index(np.argmax(acc_mat_central), acc_mat_central.shape)
    
    params_central = {'epsilon':epsilon_vec[eps_central], 'kappa':kappa_vec[kap_central], 'pnorm':1}
    
    classifier_central = centralized_classifier(params_central)
    w_opt_central = classifier_central.train(central_set)
    y_pred_central, acc_central = classifier_central.test(test_set)
    acc_vec_central[i] = acc_central

    
    for m in range(len(max_iter_vec)):
        for p in range(len(param_vec)):
            for k in range(n_folds):

                params_fedAvg = {'epsilon':epsilon_vec_algs, 'batch_size':batch_size, 'local_steps':local_steps, 
                                'max_iter':max_iter_vec[m], 'lr':param_vec[p]}
                init_vars_fedAvg = {'w':np.zeros(P)}
                
                params_fedSGD = {'epsilon':epsilon_vec_algs, 'batch_size':batch_size, 'local_steps':local_steps, 
                             'max_iter':max_iter_vec[m], 'lr':param_vec[p]}
                init_vars_fedSGD = {'w':np.zeros(P)}
                
                params_fedProx = {'epsilon':epsilon_vec_algs, 'batch_size':batch_size, 'local_steps':local_steps, 
                                'max_iter':max_iter_vec[m], 'lr':param_vec[p], 'mu': mu}
                init_vars_fedProx = {'w':np.zeros(P)}
                
                params_fedDRO_kl = {'lr':param_vec[p], 'batch_size':batch_size, 'local_steps':local_steps,
                            'max_iter':max_iter_vec[m], 'lam':1e2, 'beta':0.1*param_vec[p]}
                init_vars_fedDRO_kl = {'w':np.zeros(P), 'z':0}
                
                params_ADMM = {'epsilon':epsilon_vec_algs, 'kappa':np.ones(G)*1, 'pnorm':1,
                           'rho':param_vec[p], 'max_iter':max_iter_vec[m], 'tau':0}
                init_vars_ADMM = {'w':np.zeros(P), 'mu':np.ones([G,P])}
                
                params_ADMM_sc = {'epsilon':epsilon_vec_algs, 'kappa':np.ones(G)*1, 'pnorm':1,
                           'rho':param_vec[p], 'max_iter':max_iter_vec[m], 'tau':18*param_vec[p]}
                init_vars_ADMM_sc = {'w':np.zeros(P), 'mu':np.ones([G,P])}
                
                params_subgrad = {'epsilon':epsilon_vec_algs, 'kappa':np.ones(G)*1, 'pnorm':float('Inf'), 
                              'max_iter':max_iter_vec_subgrad[m], 'stepsize':param_vec_subgrad[p]}
                init_vars_subgrad = {'w':np.zeros(P)}
                
                client_sets_curr = {}
                client_sets_curr_norm = {}
                val_curr = {}
                val_curr_norm = {}
                
                for g2 in range(G):
                    client_sets_curr['x'+str(g2)] = client_sets_split['x'+str(g2)+'_'+str(k)]
                    client_sets_curr['y'+str(g2)] = client_sets_split['y'+str(g2)+'_'+str(k)]
                    
                    client_sets_curr_norm['x'+str(g2)] = client_sets_norm_split['x'+str(g2)+'_'+str(k)]
                    client_sets_curr_norm['y'+str(g2)] = client_sets_norm_split['y'+str(g2)+'_'+str(k)]
                    
                    val_curr['x'] = val_sets['x_split'+str(k)]
                    val_curr['y'] = val_sets['y_split'+str(k)]
                    
                    val_curr_norm['x'] = val_sets_norm['x_split'+str(k)]
                    val_curr_norm['y'] = val_sets_norm['y_split'+str(k)]



                classifier_fedAvg = fedAvg(params_fedAvg, init_vars_fedAvg, Ng_vec)
                w_opt_fedAvg = classifier_fedAvg.train(client_sets_curr)
                y_pred_fedAvg, acc_fedAvg = classifier_fedAvg.test(val_curr)
                acc_tensor_fedAvg[m,p,k] = acc_fedAvg
                
                classifier_fedSGD = fedSGD(params_fedSGD, init_vars_fedSGD, Ng_vec)
                w_opt_fedSGD = classifier_fedSGD.train(client_sets_curr)
                y_pred_fedSGD, acc_fedSGD = classifier_fedSGD.test(val_curr)
                acc_tensor_fedSGD[m,p,k] = acc_fedSGD
                
                classifier_fedProx = fedProx(params_fedProx, init_vars_fedProx, Ng_vec)
                w_opt_fedProx = classifier_fedProx.train(client_sets_curr)
                y_pred_fedProx, acc_fedProx = classifier_fedProx.test(val_curr)
                acc_tensor_fedProx[m,p,k] = acc_fedProx
                
                classifier_fedDRO_kl = fedDRO_kl(params_fedDRO_kl, init_vars_fedDRO_kl, Ng_vec)
                w_opt_fedDRO_kl = classifier_fedDRO_kl.train(client_sets_curr)
                y_pred_fedDRO_kl, acc_fedDRO_kl = classifier_fedDRO_kl.test(val_curr)
                acc_tensor_fedDRO_kl[m,p,k] = acc_fedDRO_kl
                
                classifier_ADMM = ADMM_alg(params_ADMM, init_vars_ADMM, Ng_vec)
                w_opt_ADMM,adj_ADMM = classifier_ADMM.train(client_sets_curr)
                y_pred_ADMM, acc_ADMM = classifier_ADMM.test(val_curr)
                acc_tensor_ADMM[m,p,k] = acc_ADMM
                
                classifier_ADMM_sc = ADMM_alg(params_ADMM_sc, init_vars_ADMM_sc, Ng_vec)
                w_opt_ADMM_sc,adj_ADMM_sc = classifier_ADMM_sc.train(client_sets_curr)
                y_pred_ADMM_sc, acc_ADMM_sc = classifier_ADMM_sc.test(val_curr)
                acc_tensor_ADMM_sc[m,p,k] = acc_ADMM_sc
                
                classifier_subgrad = subgradient_alg(params_subgrad, init_vars_subgrad, Ng_vec)
                w_opt_subgrad,adj_subgrad = classifier_subgrad.train(client_sets_curr_norm)
                y_pred_subgrad, acc_subgrad = classifier_subgrad.test(val_curr_norm)
                acc_tensor_SM[m,p,k] = acc_subgrad
                
    acc_mat_fedAvg = np.mean(acc_tensor_fedAvg,axis=2)
    acc_mat_fedSGD = np.mean(acc_tensor_fedSGD,axis=2)
    acc_mat_fedProx = np.mean(acc_tensor_fedProx,axis=2)
    acc_mat_fedDRO_kl = np.mean(acc_tensor_fedDRO_kl,axis=2)
    acc_mat_ADMM = np.mean(acc_tensor_ADMM,axis=2)
    acc_mat_ADMM_sc = np.mean(acc_tensor_ADMM_sc,axis=2)
    acc_mat_SM = np.mean(acc_tensor_SM,axis=2)
    
    m_fedAvg, p_fedAvg = np.unravel_index(np.argmax(acc_mat_fedAvg), acc_mat_fedAvg.shape)
    m_fedSGD, p_fedSGD = np.unravel_index(np.argmax(acc_mat_fedSGD), acc_mat_fedSGD.shape)
    m_fedProx, p_fedProx = np.unravel_index(np.argmax(acc_mat_fedProx), acc_mat_fedProx.shape)
    m_fedDRO_kl, p_fedDRO_kl = np.unravel_index(np.argmax(acc_mat_fedDRO_kl), acc_mat_fedDRO_kl.shape)
    m_ADMM, p_ADMM = np.unravel_index(np.argmax(acc_mat_ADMM), acc_mat_ADMM.shape)
    m_ADMM_sc, p_ADMM_sc = np.unravel_index(np.argmax(acc_mat_ADMM_sc), acc_mat_ADMM_sc.shape)
    m_SM, p_SM = np.unravel_index(np.argmax(acc_mat_SM), acc_mat_SM.shape)
    
    params_fedAvg = {'epsilon':epsilon_vec_algs, 'batch_size':batch_size, 'local_steps':local_steps, 
                                'max_iter':max_iter_vec[m_fedAvg], 'lr':param_vec[p_fedAvg]}
    init_vars_fedAvg = {'w':np.zeros(P)}
    
    params_fedSGD = {'epsilon':epsilon_vec_algs, 'batch_size':batch_size, 'local_steps':local_steps, 
                             'max_iter':max_iter_vec[m_fedSGD], 'lr':param_vec[p_fedSGD]}
    init_vars_fedSGD = {'w':np.zeros(P)}
    
    params_fedProx = {'epsilon':epsilon_vec_algs, 'batch_size':batch_size, 'local_steps':local_steps, 
                    'max_iter':max_iter_vec[m_fedProx], 'lr':param_vec[p_fedProx], 'mu': mu}
    init_vars_fedProx = {'w':np.zeros(P)}
    
    params_fedDRO_kl = {'lr':param_vec[p_fedDRO_kl], 'batch_size':batch_size, 'local_steps':local_steps,
                            'max_iter':max_iter_vec[m_fedDRO_kl], 'lam':1e2, 'beta':0.1*param_vec[p_fedDRO_kl]}
    init_vars_fedDRO_kl = {'w':np.zeros(P), 'z':0}
    
    params_ADMM = {'epsilon':epsilon_vec_algs, 'kappa':np.ones(G)*1, 'pnorm':1,
                            'rho':param_vec[p_ADMM], 'max_iter':max_iter_vec[m_ADMM], 'tau':0}
    init_vars_ADMM = {'w':np.zeros(P), 'mu':np.ones([G,P])}
    
    params_ADMM_sc = {'epsilon':epsilon_vec_algs, 'kappa':np.ones(G)*1, 'pnorm':1,
                            'rho':param_vec[p_ADMM_sc], 'max_iter':max_iter_vec[m_ADMM_sc], 'tau':0}
    init_vars_ADMM_sc = {'w':np.zeros(P), 'mu':np.ones([G,P])}
    
    params_subgrad = {'epsilon':epsilon_vec_algs, 'kappa':np.ones(G)*1, 'pnorm':float('Inf'), 
                              'max_iter':max_iter_vec_subgrad[m_SM], 'stepsize':param_vec_subgrad[p_SM]}
    init_vars_subgrad = {'w':np.zeros(P)}
    
    classifier_fedAvg = fedAvg(params_fedAvg, init_vars_fedAvg, Ng_vec)
    w_opt_fedAvg = classifier_fedAvg.train(client_sets)
    y_pred_fedAvg, acc_fedAvg = classifier_fedAvg.test(test_set)
    acc_vec_fedAvg[i] = acc_fedAvg
    
    classifier_fedSGD = fedSGD(params_fedSGD, init_vars_fedSGD, Ng_vec)
    w_opt_fedSGD = classifier_fedSGD.train(client_sets)
    y_pred_fedSGD, acc_fedSGD = classifier_fedSGD.test(test_set)
    acc_vec_fedSGD[i] = acc_fedSGD
    
    classifier_fedProx = fedProx(params_fedProx, init_vars_fedProx, Ng_vec)
    w_opt_fedProx = classifier_fedProx.train(client_sets)
    y_pred_fedProx, acc_fedProx = classifier_fedProx.test(test_set)
    acc_vec_fedProx[i] = acc_fedProx
    
    classifier_fedDRO_kl = fedDRO_kl(params_fedDRO_kl, init_vars_fedDRO_kl, Ng_vec)
    w_opt_fedDRO_kl = classifier_fedDRO_kl.train(client_sets)
    y_pred_fedDRO_kl, acc_fedDRO_kl = classifier_fedDRO_kl.test(test_set)
    acc_vec_fedDRO_kl[i] = acc_fedDRO_kl
    
    classifier_ADMM = ADMM_alg(params_ADMM, init_vars_ADMM, Ng_vec)
    w_opt_ADMM,adj_ADMM = classifier_ADMM.train(client_sets)
    y_pred_ADMM, acc_ADMM = classifier_ADMM.test(test_set)
    acc_vec_ADMM[i] = acc_ADMM
    
    classifier_ADMM_sc = ADMM_alg(params_ADMM_sc, init_vars_ADMM_sc, Ng_vec)
    w_opt_ADMM_sc,adj_ADMM_sc = classifier_ADMM_sc.train(client_sets)
    y_pred_ADMM_sc, acc_ADMM_sc = classifier_ADMM_sc.test(test_set)
    acc_vec_ADMM_sc[i] = acc_ADMM_sc
    
    classifier_subgrad = subgradient_alg(params_subgrad, init_vars_subgrad, Ng_vec)
    w_opt_subgrad,adj_subgrad = classifier_subgrad.train(client_sets_norm)
    y_pred_subgrad, acc_subgrad = classifier_subgrad.test(test_set_norm)
    acc_vec_SM[i] = acc_subgrad
            
            
save_exp_file(P,G,perc_train,perc_test,perc_g_vec,num_exp,max_iter_vec,
                 param_vec,acc_vec_fedAvg,acc_vec_fedSGD,acc_vec_fedProx,
                 acc_vec_fedDRO_kl,acc_vec_SM,acc_vec_ADMM,acc_vec_ADMM_sc,
                 acc_vec_central,batch_size,local_steps,mu)