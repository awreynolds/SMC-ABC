### SMC-ABC REJECTION ALGORITHM ###
### Python 3.6 ###


#Import packages
import msprime
import time
import signal
import math
import numpy as np
import pandas
import allel
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import time
from tqdm import tqdm
import subprocess
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
import rpy2.robjects.packages as rpackages
import rpy2.robjects.numpy2ri
import pandas as pd
from statistics import mean
import scipy.stats

#import R package for working with truncated multivariate gaussian
tmvtnorm = rpackages.importr('tmvtnorm')
base = rpackages.importr('base')
rtmvnorm = robjects.r['rtmvnorm']
dtmvnorm = robjects.r['dtmvnorm']

#Define Important file names
recomb_file = "/home/austin/Desktop/rejector_xaltocan/chr22_mapforsim.txt"
eroh_file = "/home/austin/Desktop/rejector_xaltocan/testchr22_ROH.hom"
sroh_file = "/home/austin/Desktop/rejector_xaltocan/testchr22_ROH.log"

chromosome_data = """\
chr1 	 249250621 	 1.1485597641285933e-08
chr2 	 243199373 	 1.1054289277533446e-08
chr3 	 198022430 	 1.1279585624662551e-08
chr4 	 191154276 	 1.1231162636001008e-08
chr5 	 180915260 	 1.1280936570022824e-08
chr6 	 171115067 	 1.1222852661225285e-08
chr7 	 159138663 	 1.1764614397655721e-08
chr8 	 146364022 	 1.1478465778920576e-08
chr9 	 141213431 	 1.1780701596308656e-08
chr10 	 135534747 	 1.3365134257075317e-08
chr11 	 135006516 	 1.1719334320833283e-08
chr12 	 133851895 	 1.305017186986983e-08
chr13 	 115169878 	 1.0914860554958317e-08
chr14 	 107349540 	 1.119730771394731e-08
chr15 	 102531392 	 1.3835785893339787e-08
chr16 	 90354753 	 1.4834607113882717e-08
chr17 	 81195210 	 1.582489036239487e-08
chr18 	 78077248 	 1.5075956950023575e-08
chr19 	 59128983 	 1.8220141872466202e-08
chr20 	 63025520 	 1.7178269031631664e-08
chr21 	 48129895 	 1.3045214034879191e-08
chr22 	 51304566 	 1.4445022767788226e-08
chrX 	 155270560 	 1.164662223273842e-08
chrY 	 59373566 	 0.0
"""

#Define important variables
len_chr = 51304566 #length of empirical chromosome
chr_rec_rate = 1.45e-8 #average recombination rate across empirical chromosome
#large_unif_low=100 #assign lower bound of uniform distribution for before/after bottleneck
#small_unif_low=10 #assign lower bound of uniform distribution for during bottleneck
#unif_high=10000 #assign higher bound of uniform distribution for before/after bottleneck
#thigh_unif=25 #assign higher bound of uniform distribution for bottleneck start/end
#tlow_unif=4 #assign lower bound of the uniform distriburion for bottleneck start/end
#delta = 0.1 #assign acceptance window
numsims = 1000 #define number of simulations to accept
numsamples = 41 #assign number of diploid individuals in your empirical sample

#start a timer for the run
start_time = time.time()

#Read in recombination map (OPTIONAL)
recomb_map = msprime.RecombinationMap.read_hapmap(recomb_file)

#Define the demographic model for msprime
def austin(preNE=10000,NE=100,postNE=5000,Nsamples=20,Tend=10,Tstart=25,debug=False):
    # population sizes
    N_Xpostb = postNE
    N_Xb = NE
    N_Xpreb = preNE
    # event times in generations
    T_Xb = Tend
    T_Xpreb = Tstart
    # the growth rate
    r_X = 0
    # initialize populations
    population_configurations = [
        msprime.PopulationConfiguration(
            sample_size=Nsamples, initial_size=N_Xpostb, growth_rate=r_X)
    ]
    migration_matrix = [
        [0]
    ]
    demographic_events = [
        #start bottleneck
        msprime.PopulationParametersChange(
            time=T_Xb, initial_size=N_Xb, population_id=0),
        #end bottleneck
        msprime.PopulationParametersChange(
            time=T_Xpreb, initial_size=N_Xpreb, population_id=0)
    ]
    if debug:
        dd = msprime.DemographyDebugger(
            population_configurations=population_configurations,
            migration_matrix=migration_matrix,
            demographic_events=demographic_events)
        dd.print_history()
        return
    return population_configurations, migration_matrix, demographic_events

#Calculate fROH from empirical data
rread_command = "read.table(file = \"" + eroh_file + "\", header = T)"
r_df = robjects.r(rread_command)
pd_df = pd.DataFrame.from_dict({ key : np.asarray(r_df.rx2(key)) for key in r_df.names })
#calculate croh for each individual
ecroh_pd_df = pd_df.groupby("IID").sum()
#calc avg fROH for simulated data
eFROH = mean(ecroh_pd_df['KB']*1000/len_chr)
print("Empirical FROH is " + str(eFROH))

#Find the number of variants in empirical data
for line in open(sroh_file):
 if "out of" in line:
    evariants = int(line.split(" ", 1)[0])
    print("%d variants in empirical dataset" % (evariants))

#define variables for the prior distribution
prior_mean = [np.log(200),np.log(500),np.log(400),np.log(15),np.log(20)]
m = [2,0,0,0,0,
       0,2,0,0,0,
       0,0,2,0,0,
       0,0,0,1,0,
       0,0,0,0,1]
prior_sigma = robjects.r['matrix'](base.as_numeric(m), nrow=5, byrow="TRUE")

d = [1,0,0,0,0,
    -1,1,0,0,0,
    -1,0,1,0,0,
    0,0,0,1,0,
    0,0,0,-1,1]
D_constraints = base.matrix(base.as_numeric(d), nrow=5, byrow="TRUE")
#print(D_constraints)
#print(prior_sigma)
lower_bounds = [float("-inf"), 0, 0, float("-inf"), 0]
upper_bounds = [float("inf"),float("inf"),float("inf"),float("inf"),float("inf")]

#assign time limit for simulations
n_minutes = 60*60 #one hour
#make signal handler for timing out
def signal_handler(signum, frame):
    raise Exception("Timed out!")

#SMC algorithm
#define thresholds and start weight variable
thresholds = [0.2,0.1,0.05,0.01]
start_weights = []
accepted_sims = [] #empty list for accepted parameter values
listodists = [] #empty list to fill with accepted distances

#first iteration
for i in tqdm(range(1,numsims+1)):
    #print("for-loop iteration %d" % (i))
    distance = thresholds[0] + 0.1 #assign starting distance
    counter = 0
    while distance>thresholds[0]:
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(n_minutes)
        try:
            #print("starting distance: %f" % (distance)) #print starting distance

            #Choose starting parameters from a truncated multivariate gaussian
            #the values are log transformed
            candidate_logparameters = rtmvnorm(n=1,
                  mean = robjects.FloatVector(prior_mean),
                  sigma = prior_sigma,
                  lower= robjects.FloatVector(lower_bounds),
                  upper= robjects.FloatVector(upper_bounds),
                  D = D_constraints,
                  algorithm="gibbs")
            #print(candidate_logparameters)

            #exponentiate the values from the multivariate normal dist for use in msprime
            candidate_parameters = base.exp(candidate_logparameters)
            #print("Chosen candidate parameter values: ")
            #print(candidate_parameters)

            #define model
            pc, mm, de = austin(preNE=candidate_parameters[1],NE=candidate_parameters[0],postNE=candidate_parameters[2],Nsamples=numsamples*2,Tstart=candidate_parameters[4],Tend=candidate_parameters[3])

            #run simulation with generated parameters
            #using average recombination rate over length of chromosome method
            #print("simulating... " + time.asctime( time.localtime(time.time()) ))
            #ts = msprime.simulate(length=len_chr,
            #                      recombination_rate=chr_rec_rate,
            #                      mutation_rate=1e-8,
            #                      population_configurations=pc,
            #                      migration_matrix=mm,
            #                      demographic_events=de)
            #print("simulation complete " + time.asctime( time.localtime(time.time()) ))
            #or using the recombination map method
            print("simulating... " + time.asctime( time.localtime(time.time()) ))
            ts = msprime.simulate(recombination_map=recomb_map,
                                  mutation_rate=1e-8,
                                  population_configurations=pc,
                                  migration_matrix=mm,
                                  demographic_events=de)
            print("simulation complete " + time.asctime( time.localtime(time.time()) ))
            #write treesequence to vcf
            with open("/home/austin/Desktop/rejector_xaltocan/simulated.vcf", "w") as vcf_file:
                ts.write_vcf(vcf_file, 2)

            #fix simulated sample names
            subprocess.call("sed -i 's/msp_0/msp_00/g' /home/austin/Desktop/rejector_xaltocan/simulated.vcf", shell=True)

            #downsample to same number of variants as in empirical data
            #command = "(cat /home/austin/Desktop/rejector_xaltocan/simulated.vcf | head -n 10000 | grep ^# ; grep -v ^# /home/austin/Desktop/rejector_xaltocan/simulated.vcf | shuf -n "+ str(evariants) +" | LC_ALL=C sort -k1,1V -k2,2n) > /home/austin/Desktop/rejector_xaltocan/downsampled_simulated.vcf"
            #old command ^ not sure why head was there?
            command = "(cat /home/austin/Desktop/rejector_xaltocan/simulated.vcf | grep ^# ; grep -v ^# /home/austin/Desktop/rejector_xaltocan/simulated.vcf | shuf -n "+ str(evariants) +" | LC_ALL=C sort -k1,1V -k2,2n) > /home/austin/Desktop/rejector_xaltocan/downsampled_simulated.vcf"
            subprocess.call(command, shell=True)

            #calculate ROH for the simulated data
            subprocess.call("~/Desktop/bin/plink_1.9/plink --vcf /home/austin/Desktop/rejector_xaltocan/downsampled_simulated.vcf --silent --homozyg --homozyg-snp 50 --homozyg-window-missing 2 --homozyg-window-het 1 --homozyg-kb 500 --out /home/austin/Desktop/rejector_xaltocan/downsampled_simulated_ROH", shell=True)

            #read file back into python
            r_df = robjects.r('read.table(file = "/home/austin/Desktop/rejector_xaltocan/downsampled_simulated_ROH.hom", header = T)')
            pd_df = pd.DataFrame.from_dict({ key : np.asarray(r_df.rx2(key)) for key in r_df.names })

            if pd_df.empty:
                distance = thresholds[0] + 0.1
                counter = counter + 1
            else:
                #calculate croh for each individual
                croh_pd_df = pd_df.groupby("IID").sum()
                #calc avg fROH for simulated data
                sFROH = mean(croh_pd_df['KB']*1000/len_chr)
                print("Simulated FROH: %f" % (sFROH)) #print simulated FROH

                #calc distance between simulated avgfROH and actual avgrROH
                distance=abs(eFROH-sFROH)
                print("new distance: %f" % (distance)) #print new distance

                counter = counter + 1
        except (Exception):
            print("Simulation took longer than 1 hour! Resampling...")
        signal.alarm(0)
    #print number of accepted sims
    print("%d total simulations for this iteration" % (counter))
    #append parameter values from accepted sims
    accepted_sims.append([candidate_parameters[1],candidate_parameters[0],candidate_parameters[2],candidate_parameters[4],candidate_parameters[3]])
    start_weights.append(1/(numsims))
    listodists.append(distance)
    #print("%d accepted simulations so far." % (len(accepted_sims)))
    #print(accepted_sims)


#define the covariance matrix of the accepted parameter combinations, sigma
sigmat = 2*np.cov(np.log(np.array(accepted_sims).T))
rpy2.robjects.numpy2ri.activate()
nr,nc = sigmat.shape
Br = robjects.r.matrix(sigmat, nrow=nr, ncol=nc)
robjects.r.assign("sigmat", Br)
#print("sigmat matrix")
#print(sigmat)
rpy2.robjects.numpy2ri.deactivate()
#print(listodists)
#write accepted parameters to file
np.savetxt(fname="/home/austin/Desktop/rejector_xaltocan/recombmap_chr22_accepted_parameters_t{}.txt".format(thresholds[0]),X=accepted_sims,delimiter="\t")
#write weights of accepted parameters to file
np.savetxt(fname="/home/austin/Desktop/rejector_xaltocan/recombmap_chr22_weights_t{}.txt".format(thresholds[0]),X=start_weights,delimiter="\t")

for t in tqdm(range(1,len(thresholds))):
    print("New threshold: " + str(thresholds[t]))
    new_accepted_sims = []
    new_start_weights = []
    weights_denom = []
    for i in tqdm(range(1,numsims+1)):
        counter = 0
        distance = thresholds[t] + 0.1 #assign starting distance
        while distance>thresholds[t]:
            signal.signal(signal.SIGALRM, signal_handler)
            signal.alarm(n_minutes)
            try:
                #choose a line of previously accepted results at random
                choice_indices = np.random.choice(len(accepted_sims), 1, p=start_weights)
                choices = [accepted_sims[j] for j in choice_indices][0]
                #print(choices)
                ##from that line create a multivariate gaussian distribution with mean = chosen line and variance = sigma1
                rpy2.robjects.numpy2ri.activate()#activate the numpy-r convertor
                #convert to log scale for input into rtmvnorm
                candidate_logparameters = rtmvnorm(n=1,
                      mean = robjects.FloatVector(np.log(choices)),
                      sigma = sigmat,
                      lower= robjects.FloatVector(lower_bounds),
                      upper= robjects.FloatVector(upper_bounds),
                      D = D_constraints,
                      algorithm="gibbs")
                #print(candidate_logparameters)
                rpy2.robjects.numpy2ri.deactivate()#deactivate the numpy r convertor so that pandas will work below

                #exponentiate the values from the multivariate normal dist for use in msprime
                candidate_parameters = base.exp(candidate_logparameters)
                #print("Chosen candidate parameter values: ")
                #print(candidate_parameters)

                #define model
                pc, mm, de = austin(preNE=candidate_parameters[1],NE=candidate_parameters[0],postNE=candidate_parameters[2],Nsamples=numsamples*2,Tstart=candidate_parameters[4],Tend=candidate_parameters[3])            #run simulation with generated parameters
                #using average recombination rate over length of chromosome method
                #print("simulating... " + time.asctime( time.localtime(time.time()) ))
                #ts = msprime.simulate(length=len_chr,
                #                  recombination_rate=chr_rec_rate,
                #                  mutation_rate=1e-8,
                #                  population_configurations=pc,
                #                  migration_matrix=mm,
                #                  demographic_events=de)
                #print("simulation complete " + time.asctime( time.localtime(time.time()) ))
                #or using the recombination map method
                print("simulating... " + time.asctime( time.localtime(time.time()) ))
                ts = msprime.simulate(recombination_map=recomb_map,
                                      mutation_rate=1e-8,
                                      population_configurations=pc,
                                      migration_matrix=mm,
                                      demographic_events=de)
                print("simulation complete " + time.asctime( time.localtime(time.time()) ))
                #write treesequence to vcf
                with open("/home/austin/Desktop/rejector_xaltocan/simulated.vcf", "w") as vcf_file:
                    ts.write_vcf(vcf_file, 2)
                #fix simulated sample names
                subprocess.call("sed -i 's/msp_0/msp_00/g' /home/austin/Desktop/rejector_xaltocan/simulated.vcf", shell=True)
                #downsample to same number of variants as in empirical data
                #command = "(cat /home/austin/Desktop/rejector_xaltocan/simulated.vcf | head -n 10000 | grep ^# ; grep -v ^# /home/austin/Desktop/rejector_xaltocan/simulated.vcf | shuf -n "+ str(evariants) +" | LC_ALL=C sort -k1,1V -k2,2n) > /home/austin/Desktop/rejector_xaltocan/downsampled_simulated.vcf"
                #old command ^ not sure why the head command was there?
                command = "(cat /home/austin/Desktop/rejector_xaltocan/simulated.vcf | grep ^# ; grep -v ^# /home/austin/Desktop/rejector_xaltocan/simulated.vcf | shuf -n "+ str(evariants) +" | LC_ALL=C sort -k1,1V -k2,2n) > /home/austin/Desktop/rejector_xaltocan/downsampled_simulated.vcf"
                subprocess.call(command, shell=True)

                #calculate ROH for the simulated data
                subprocess.call("~/Desktop/bin/plink_1.9/plink --vcf /home/austin/Desktop/rejector_xaltocan/downsampled_simulated.vcf --silent --homozyg --homozyg-snp 50 --homozyg-window-missing 2 --homozyg-window-het 1 --homozyg-kb 500 --out /home/austin/Desktop/rejector_xaltocan/downsampled_simulated_ROH", shell=True)
                #read file back into python
                r_df = robjects.r('read.table(file = "/home/austin/Desktop/rejector_xaltocan/downsampled_simulated_ROH.hom", header = T)')
                pd_df = pd.DataFrame.from_dict({ key : np.asarray(r_df.rx2(key)) for key in r_df.names })
                if pd_df.empty:
                    distance = thresholds[t] + 0.1
                    counter = counter + 1
                else:
                    #calculate croh for each individual
                    croh_pd_df = pd_df.groupby("IID").sum()
                    #calc avg fROH for simulated data
                    sFROH = mean(croh_pd_df['KB']*1000/len_chr)
                    print("Simulated FROH: %f" % (sFROH)) #print simulated FROH
                    #calc distance between simulated avgfROH and actual avgrROH
                    distance=abs(eFROH-sFROH)
                    print("new distance: %f" % (distance)) #print new distance
                    counter = counter + 1
            except (Exception):
                print("Simulation took longer than 1 hour! Resampling...")
            signal.alarm(0)
        #print number of accepted sims
        print("%d total simulations for this iteration" % (counter))
        #append parameter values from accepted sims
        new_accepted_sims.append([candidate_parameters[1],candidate_parameters[0],candidate_parameters[2],candidate_parameters[4],candidate_parameters[3]])
        #we are calculating the new weights "new_start_weights
        #find the density of the candidate parameters in multidimensional distribution
        #based on the previous iteration's accepted parameter sets
        for sim in range(0,numsims):
            theta_sim = dtmvnorm(x=base.as_vector(candidate_logparameters),
                                 mean = robjects.FloatVector(np.log(accepted_sims[sim])),
                                 sigma = Br,
                                 log = "TRUE")
            weights_denom.append(np.log(start_weights[sim])+np.array(theta_sim[0]))
        #print(weights_denom)
        #find largest value in weights denom matrix
        lmax = np.amax(weights_denom)
        #subtract lmax from each value in weights denom
        weights_denom_star = weights_denom-lmax
        #sum all values in weights denom star
        lsum = np.sum(np.exp(weights_denom_star))
        #calculate final denominator value as lmax + the log of lsum values
        log_q = lmax + np.log(lsum)
        #print(log_q)
        #now making the numerator of the weights equation
        #find the density of the candidate parameters in multidimensional distribution
        #based on the original prior values
        log_p = dtmvnorm(x=base.as_vector(candidate_logparameters),
                            mean = robjects.FloatVector(prior_mean),
                            sigma = prior_sigma,
                            log = "TRUE")
        #print(log_p)
        #calculate the weight for the i-th new accepted parameter set
        new_start_weight = np.exp(log_p - log_q)
        #print(new_start_weight)
        #append new weight to new_start_weights
        new_start_weights.append(new_start_weight[0])
        #print("%d accepted simulations so far." % (len(new_accepted_sims)))
        #print(new_accepted_sims)
        listodists.append(distance)
    #print("old sims")
    #print(accepted_sims)
    #print("new sims")
    #print(new_accepted_sims)
    #print("old weights")
    #print(start_weights)


    #assign new accepted sims to accepted sims for next threshold loop iteration
    accepted_sims = new_accepted_sims

    #once all new start weights are collected
    #need to turn them into probabilities
    #by dividing each one by the sum of the weights
    sum_weights = np.sum(new_start_weights)

    #now assign new start weights to start weights as a probability for next threshold loop iteration
    start_weights = new_start_weights/sum_weights
    #print("new weights")
    #print(start_weights)
    #print("old sigmas")
    #print(sigmat)
    #calculate sigmat for next threshold loop iteration
    sigmat = 2*np.cov(np.log(np.array(accepted_sims).T))
    #print("new sigmas")
    #print(sigmat)
    nr,nc = sigmat.shape
    rpy2.robjects.numpy2ri.activate()#activate the numpy-r convertor
    Br = robjects.r.matrix(sigmat, nrow=nr, ncol=nc)
    robjects.r.assign("sigmat", Br)
    rpy2.robjects.numpy2ri.deactivate()#deactivate the numpy-r convertor

    #write accepted parameters to file
    np.savetxt(fname="/home/austin/Desktop/rejector_xaltocan/recombmap_chr22_accepted_parameters_t{}.txt".format(thresholds[t]),X=accepted_sims,delimiter="\t")
    #write weights of accepted parameters to file
    np.savetxt(fname="/home/austin/Desktop/rejector_xaltocan/recombmap_chr22_weights_t{}.txt".format(thresholds[t]),X=start_weights,delimiter="\t")

#write accepted distances to file
np.savetxt(fname="/home/austin/Desktop/rejector_xaltocan/recombmap_chr22_distances.txt",X=listodists,delimiter="\t")

endtime = time.time() - start_time
#print runtime
print("SMC-ABC run complete! This run took " + str(endtime) + " seconds.")
