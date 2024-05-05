import numpy as np
import random
from sklearn.model_selection import train_test_split
from methods.baseline import Baseline
from utils import compute_metrics

class FastMIML(Baseline):

    def __init__(self, config):
        super(FastMIML, self).__init__(is_pytorch_model=False)
        
        ## parameters
        self.D = config.D # dimension of the shared space
        self.norm_up = config.norm_up # norm of each vector (upper bound for each norm)
        self.maxiter = config.maxiter# number of iterations
        self.step_size = config.step_size  # step size of SGD
        self.lambda_reg = config.lambda_reg # regularization parameter        
        self.num_sub = config.num_sub # number of sub concepts
    
    def FastMIML_train(self, train_data, train_targets, W, V, costs, norm_up, step_size0, num_sub, AW, AV, Anum, trounds, lambd, opts):
        average_begin = opts['average_begin']
        average_size = opts['average_size']
        n, n_class = train_targets.shape
        # tmpnums: a vector consisting of no of labels that are relevant in each bags
        tmpnums = np.sum(train_targets >= 1, axis=1)
        # Initialize an empty array to hold the pairs
        train_pairs = []

        # Loop through each row of train_targets
        for i in range(n):
            # Get the indices of all elements in the row that are greater than or equal to 1
            idx = np.where(train_targets[i,:] >= 1)[0]
            # Loop through each index(present in idx) and add the (i, j) pair to train_pairs
            for j in idx:
                train_pairs.append((i, j))
        # Convert train_pairs to a numpy array
        train_pairs = np.array(train_pairs)
        n_pairs = len(train_pairs)
        random_idx = np.random.permutation(n_pairs) #generates random nos array in range n_pairs(1147)
        trounds = int(trounds)

        for i in range(n_pairs):
            '''We are storing the index value of random bags one by one which we obtain from the random_idx '''
            idx_ins = int(train_pairs[random_idx[i]][0])
    #         print('idx_ins',idx_ins)
            '''This is how we load the instances present in each bag:'''
    #         xbag = train_data[idx_ins]
            xbag = train_data[:][idx_ins][0]
            '''This is how we load the label as well for each random bags one by one since loop
            will take onebyone'''
            idx_class = int(train_pairs[random_idx[i]][1])
            '''Now lets go for irrelavent labels:'''
            '''if idx_class=dummy_variable(2),then idx_irr contains all the irrelevant rows only
            otherwise it contains dummy_variable as well as all irrelevant variables'''
            if idx_class == n_class:
                idx_irr = np.where(train_targets[idx_ins, :] <= 0)[0]
            else:
                idx_irr = np.where(train_targets[idx_ins, :] != 1)[0]
            n_irr = len(idx_irr)


            Wy=W[:, idx_class*num_sub : (idx_class+1)*num_sub] 
    #         print('shape of Wy',Wy.shape)
    #         print('shape of V', V.shape)

            Vbag = V @ xbag
    #         print('shape of Vbag: ',Vbag.shape)


            fs = np.dot(Wy.T, Vbag)
            idx_max_class = np.argmax(fs, axis=0)
            fs = np.max(fs, axis=0)

            fy = np.max(fs)

    # # #      idx_max_ins1 is a index of KEY INSTANCE for that label
            idx_max_ins1 = np.argmax(fs)


            idx_max_class = idx_max_class[idx_max_ins1]
    #         print('idx_max_class: ',idx_max_class)
            Wy = Wy[:, idx_max_class]
    #         print('shape of Wy updated: ',Wy.shape)

    #         if 1:  # two optional implementation, switch to 0 to use the matlab code
    #             print("its working")
    # #             j, idx_pick, idx_max_pick, fyn, idx_max_ins2 = sample_max1_small(n_irr, idx_irr, W, Vbag, fy, num_sub, np.random.rand())
    #         else:
            '''lets go to next step:'''
            '''Doing same things like what we did earlier for the relevant labels. see copy for explaination.'''
            for j in range(n_irr):
                idx_pick = idx_irr[random.randint(0, n_irr-1)]
                Wyn = W[:, (idx_pick)*num_sub : (idx_pick+1)*num_sub]
                fs = np.dot(Wyn.T, Vbag)
                idx_max_pick = np.argmax(fs, axis=0)
    #             print('idx_max_pick: ',idx_max_pick)
                fs = np.max(fs, axis=0)
    #             print('fs: ',fs)
                fyn = np.max(fs)
                idx_max_ins2 = np.argmax(fyn)
    #             print("idx_max_ins2:", idx_max_ins2)
                idx_max_pick = idx_max_pick[idx_max_ins2]
    #             print("idx_max_pick updated: ",idx_max_pick)
                if fyn > fy-1:
                    break

            if fyn > fy-1:  # make a gradient step, N=j;
                step_size = step_size0 / (1 + lambd * trounds * step_size0)
    #             print("step_size: ",step_size)
                trounds += 1
                Wyn = W[:, (idx_pick)*num_sub + idx_max_pick]
    #             print('shape of Wyn: ',Wyn.shape)
                loss = costs[int(np.floor(n_irr/(j+1)))]

           # ''' Now weights are updated using gradient descent and stored in tmp1'''
                tmp1 = Wy + step_size * loss * Vbag[:, idx_max_ins1]
        #         print('shape of tmp1: ',tmp1.shape)
                if opts['norm'] != 0: #This line checks whether normalization of the weight matrix is enabled.
                    tmp3 = np.linalg.norm(tmp1) #This line calculates the L2 norm of the temporary variable tmp1.
                    if tmp3 > norm_up:
                        tmp1 = tmp1 * norm_up / tmp3
        #                 print('normalized tmp1 shape: ',tmp1.shape)
                tmp2 = Wyn - step_size * loss * Vbag[:, idx_max_ins2]
                if opts['norm'] != 0:
                    tmp3 = np.linalg.norm(tmp2)
                    if tmp3 > norm_up:
                        tmp2 = tmp2 * norm_up / tmp3
        #                 print('normalized tmp2 shape: ',tmp2.shape)


                    '''Similarly we now update the V matrix:'''
        #         V = V - step_size*loss*(W[:,[(idx_pick)*num_sub+idx_max_pick, (idx_class)*num_sub+idx_max_class]] @ np.array([xbag[:,idx_max_ins2],-xbag[:,idx_max_ins1]]).T) #(Wyn*xins2'-Wy*xins1');
        #         W_subset = np.hstack(((W[:, (idx_pick)*num_sub+idx_max_pick]),( W[:, (idx_class)*num_sub+idx_max_class])))
                W_subset = np.hstack(((W[:, (idx_pick)*num_sub+idx_max_pick]), (W[:, (idx_class)*num_sub+idx_max_class]))).reshape(-1, 2)

        #         print('w_subset shape: ', W_subset.shape)
        #         xbag_subset = np.hstack((xbag[:, idx_max_ins2], -xbag[:, idx_max_ins1]))
                xbag_subset = np.hstack((xbag[:, idx_max_ins2], -xbag[:, idx_max_ins1])).reshape(2, -1)

        #         print('xbag_subset shape: ', xbag_subset.shape)
                V = V - step_size * loss * np.dot(W_subset, xbag_subset)

                '''Finally we update our weights:'''
                W[:,(idx_class)*num_sub+idx_max_class] = tmp1
                W[:,(idx_pick)*num_sub+idx_max_pick] = tmp2

                if opts['norm'] != 0:
        #             norms = np.sqrt(np.sum(V**2, axis=0))
                    norms = np.linalg.norm(V, axis=0)

                    idx_down = np.where(norms > norm_up)[0]
        #             print('shape of idx_down: ', idx_down.shape)
                    if len(idx_down) > 0:
    #                     norms = norms[norms<=norm_up]
                        norms = norms[norms > norm_up]

        #                 print('shape of norms: ', norms.shape)
                        for k in range(len(idx_down)):
                            V[:, idx_down[k]] = V[:, idx_down[k]]*norm_up/norms[k]
    #         print(' ')
    # Now AW, AV and Anum are our average weights which we will be needing in order to predict our test data.
    # AW/Anum and AV/Anum are actually average and not AW and AV
            if trounds > average_begin and i % average_size == 0:
                AW = AW + W
                AV = AV + V
                Anum = Anum + 1

        return W,V,AW,AV,Anum,trounds

    def FastMIML_test(self, test_data, W, V, num_sub):
        # Set the number of top indices to select
        firstk = 5

        # Get the number of bags
        n = len(test_data)

        # Create an empty array for storing the max score for each bag and label combination
        pres = np.full((n, int(W.shape[1]/num_sub)), -np.inf)


        # Calculate the matrix product of the transpose of V and W
        WV = V.T @ W
    #     print('shape of WV: ',WV.shape)
        # Loop over each bag in test_data and each sublabels
        for i in range(n):
    #         xbag = test_data[i]
            xbag = test_data[:][i][0]
    #         print('shape of xbag: ',xbag.shape)
            for j in range(num_sub):
                # Select every num_sub-th column of WV
                WVone = WV[:, j::num_sub]
    #             print('shape of WVone: ', WVone.shape)
                # Calculate the maximum score of each column of WVone multiplied by xbag
                fs = np.max(xbag.T @ WVone, axis=0)
    #             print('shape of fs: ',fs.shape)
                # Update the pres array with the maximum of the current pres and fs for the current bag and subset
                pres[i,:] = np.maximum(pres[i,:], fs)

        # Sort the pres array in descending order for each bag, excluding the last column
        ord = np.argsort(pres[:, :-1], axis=1)[:, ::-1]

        # Select the top firstk indices from the ord array
        top_idx = ord[:, :firstk]

        # Create an array for storing the labels
        labels = -np.ones((n, W.shape[1] // num_sub - 1))

        # Loop over each bag and set the label to 1 for the top index and any index where the pres score is greater than the last score plus 1
        for i in range(n):
            labels[i, top_idx[i, 0]] = 1
            labels[i, pres[i, :-1] > (pres[i, -1] + 1)] = 1

        # Return the pres and labels arrays
        return pres, labels

    def run(self, bags, labels):
        # train_data: n*1 cells, one cell for a bag, each cell is a n_ins*d matrix
        bags = bags.reshape(bags.shape[0], 1, bags.shape[1], -1)
        labels = labels.reshape(labels.shape[0], -1)
        # train_targets: n*n_class, one row for a bag
        

        train_data, test_data, train_targets, test_targets = train_test_split(bags, labels, test_size=0.2, random_state=42)

        opts = {'norm': 1, 'average_size': 10, 'average_begin': 0}

        ## initialization
        #creating and adding a dummy variable:
        train_targets = np.hstack((train_targets, np.ones((train_targets.shape[0], 1))*2))
        n_class = train_targets.shape[1]

        m = train_data[:][0][0].shape[0]
        costs = 1.0 / np.arange(1, n_class+1).cumsum()

    # randomly generated V and W with mean 0 and standard deviation 1/sqrt(m)
        V = np.random.normal(0, 1/np.sqrt(m), (self.D, m))  # D*m
        W = np.random.normal(0, 1/np.sqrt(m), (self.D, n_class*self.num_sub))  # D*n_class
        for k in range(m):
            tmp1 = V[:, k]
            V[:, k] = tmp1*self.norm_up/np.linalg.norm(tmp1)
        for k in range(n_class*self.num_sub):
            tmp1 = W[:, k]
            W[:, k] = tmp1*self.norm_up/np.linalg.norm(tmp1)

        AW = np.zeros((self.D, n_class*self.num_sub))
        AV = np.zeros((self.D, m))
        Anum = 0
        trounds = 0   # no of rounds in SGD

        ## train
        for i in range(self.maxiter):
            print(i)
            W, V, AW, AV, Anum, trounds = self.FastMIML_train(train_data, train_targets, W, V, costs, self.norm_up, self.step_size, self.num_sub, AW, AV, Anum, trounds, self.lambda_reg, opts)

        ## test
        test_outputs, test_labels = self.FastMIML_test(test_data, AW/Anum, AV/Anum, self.num_sub)
        precision, recall, f1 = compute_metrics(test_labels, test_targets)
        
        print(f'Epochs: {i} - Precision: {precision} - Recall: {recall} - F1: {f1}')


