### SMC-ABC REJECTION ALGORITHM ###
### Python 3.6 ###
### TO DO ###
# ADD WRITE STATEMENTS TO SAVE PARAMETER VALUES FOR EACH THRESHOLD

#Import packages
import msprime
import time
import signal
import math
import random
import os
import numpy as np
import pandas
import allel
import multiprocessing
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
set_seed = robjects.r['set.seed']

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
thresholds = [0.2,0.1,0.05,0.01] #define thresholds
numsims = 15 #define number of simulations to accept
numsamples = 41 #assign number of diploid individuals in your empirical sample
ncores = 4 #the number of cores to use in multiprocessing

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






#define function to run the first iteration of the SMC algorithm
def first_run(iters):
    start_weights = [] # empty list of start weight values
    accepted_sims = [] #empty list for accepted parameter values
    listodists = [] #empty list to fill with accepted distances
    #print("for-loop iteration %d" % (i))
    distance = thresholds[0] + 0.1 #assign starting distance
    counter = 0
    while distance>thresholds[0]:
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(n_minutes)
        try:
            #print("starting distance: %f" % (distance)) #print starting distance
            #randomization for files and seeds
            filenum = str(int(1e10*random.random()))
            set_seed(int(1e5*random.random()))
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
            print("Chosen candidate parameter values: ")
            print(candidate_parameters)

            #define model
            pc, mm, de = austin(preNE=candidate_parameters[1],NE=candidate_parameters[0],postNE=candidate_parameters[2],Nsamples=numsamples*2,Tstart=candidate_parameters[4],Tend=candidate_parameters[3])

            #run simulation with generated parameters
            #using average recombination rate over length of chromosome method
            print("simulating... " + time.asctime( time.localtime(time.time()) ))
            ts = msprime.simulate(length=len_chr,
                                  recombination_rate=chr_rec_rate,
                                  mutation_rate=1e-8,
                                  population_configurations=pc,
                                  migration_matrix=mm,
                                  demographic_events=de)
            #print("simulation complete " + time.asctime( time.localtime(time.time()) ))
            #or using the recombination map method
            #print("simulating... " + time.asctime( time.localtime(time.time()) ))
            #ts = msprime.simulate(recombination_map=recomb_map,
            #                      mutation_rate=1e-8,
            #                      population_configurations=pc,
            #                      migration_matrix=mm,
            #                      demographic_events=de)
            print("simulation complete " + time.asctime( time.localtime(time.time()) ))
            #write treesequence to vcf
            simfilename = "/home/austin/Desktop/rejector_xaltocan/simulated"+ filenum +".vcf"
            with open(simfilename, "w") as vcf_file:
                ts.write_vcf(vcf_file, 2)

            #fix simulated sample names
            writecommand = "sed -i 's/msp_0/msp_00/g' /home/austin/Desktop/rejector_xaltocan/simulated"+ filenum +".vcf"
            subprocess.call(writecommand, shell=True)

            #downsample to same number of variants as in empirical data
            #command = "(cat /home/austin/Desktop/rejector_xaltocan/simulated.vcf | head -n 10000 | grep ^# ; grep -v ^# /home/austin/Desktop/rejector_xaltocan/simulated.vcf | shuf -n "+ str(evariants) +" | LC_ALL=C sort -k1,1V -k2,2n) > /home/austin/Desktop/rejector_xaltocan/downsampled_simulated.vcf"
            #old command ^ not sure why head was there?
            command = "(cat /home/austin/Desktop/rejector_xaltocan/simulated"+ filenum +".vcf | grep ^# ; grep -v ^# /home/austin/Desktop/rejector_xaltocan/simulated"+ filenum +".vcf | shuf -n "+ str(evariants) +" | LC_ALL=C sort -k1,1V -k2,2n) > /home/austin/Desktop/rejector_xaltocan/downsampled_simulated"+ filenum +".vcf"
            subprocess.call(command, shell=True)

            #calculate ROH for the simulated data
            rohcommand = "~/Desktop/bin/plink_1.9/plink --vcf /home/austin/Desktop/rejector_xaltocan/downsampled_simulated"+ filenum +".vcf --silent --homozyg --homozyg-snp 50 --homozyg-window-missing 2 --homozyg-window-het 1 --homozyg-kb 500 --out /home/austin/Desktop/rejector_xaltocan/downsampled_simulated_ROH"+ filenum
            subprocess.call(rohcommand, shell=True)

            #read file back into python
            r_df = robjects.r('read.table(file = "/home/austin/Desktop/rejector_xaltocan/downsampled_simulated_ROH{0}.hom", header = T)'.format(filenum))
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
    accepted_sims = [candidate_parameters[1],candidate_parameters[0],candidate_parameters[2],candidate_parameters[4],candidate_parameters[3]]
    start_weights = 1/(numsims)
    listodists = distance
    #print("%d accepted simulations so far." % (len(accepted_sims)))
    #print(accepted_sims)
    removecommand = "rm /home/austin/Desktop/rejector_xaltocan/*simulated*" + filenum + ".{log,nosex,hom,hom.indiv,hom.summary,vcf}"
    subprocess.call(removecommand, shell=True, executable='/bin/bash')
    #print(accepted_sims)
    #q.put(accepted_sims)
    return(accepted_sims,start_weights,listodists)





#define function for additional thresholds
def subsequent_runs(iters):
    new_accepted_sims = []
    new_start_weights = []
    new_listodists = []
    weights_denom = []
    distance = threshold + 0.1 #assign starting distance
    counter = 0
    while distance>threshold:
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(n_minutes)
        try:
            #randomization for files and seeds
            filenum = str(int(1e10*random.random()))
            set_seed(int(1e5*random.random()))
            #choose a line of previously accepted results at random
            choice_indices = np.random.choice(len(accepted_sims_gbl), 1, p=start_weights_gbl)
            choices = [accepted_sims_gbl[j] for j in choice_indices][0]
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
            print("Chosen candidate parameter values: ")
            print(candidate_parameters)

            #define model
            pc, mm, de = austin(preNE=candidate_parameters[1],NE=candidate_parameters[0],postNE=candidate_parameters[2],Nsamples=numsamples*2,Tstart=candidate_parameters[4],Tend=candidate_parameters[3])            #run simulation with generated parameters
            #using average recombination rate over length of chromosome method
            print("simulating... " + time.asctime( time.localtime(time.time()) ))
            ts = msprime.simulate(length=len_chr,
                              recombination_rate=chr_rec_rate,
                              mutation_rate=1e-8,
                              population_configurations=pc,
                              migration_matrix=mm,
                              demographic_events=de)
            #print("simulation complete " + time.asctime( time.localtime(time.time()) ))
            #or using the recombination map method
            #print("simulating... " + time.asctime( time.localtime(time.time()) ))
            #ts = msprime.simulate(recombination_map=recomb_map,
            #                      mutation_rate=1e-8,
            #                      population_configurations=pc,
            #                      migration_matrix=mm,
            #                      demographic_events=de)
            print("simulation complete " + time.asctime( time.localtime(time.time()) ))
            #write treesequence to vcf
            filenum = str(int(1e10*random.random()))
            simfilename = "/home/austin/Desktop/rejector_xaltocan/simulated"+ filenum +".vcf"
            with open(simfilename, "w") as vcf_file:
                ts.write_vcf(vcf_file, 2)

            #fix simulated sample names
            writecommand = "sed -i 's/msp_0/msp_00/g' /home/austin/Desktop/rejector_xaltocan/simulated"+ filenum +".vcf"
            subprocess.call(writecommand, shell=True)

            #downsample to same number of variants as in empirical data
            #command = "(cat /home/austin/Desktop/rejector_xaltocan/simulated.vcf | head -n 10000 | grep ^# ; grep -v ^# /home/austin/Desktop/rejector_xaltocan/simulated.vcf | shuf -n "+ str(evariants) +" | LC_ALL=C sort -k1,1V -k2,2n) > /home/austin/Desktop/rejector_xaltocan/downsampled_simulated.vcf"
            #old command ^ not sure why head was there?
            command = "(cat /home/austin/Desktop/rejector_xaltocan/simulated"+ filenum +".vcf | grep ^# ; grep -v ^# /home/austin/Desktop/rejector_xaltocan/simulated"+ filenum +".vcf | shuf -n "+ str(evariants) +" | LC_ALL=C sort -k1,1V -k2,2n) > /home/austin/Desktop/rejector_xaltocan/downsampled_simulated"+ filenum +".vcf"
            subprocess.call(command, shell=True)

            #calculate ROH for the simulated data
            rohcommand = "~/Desktop/bin/plink_1.9/plink --vcf /home/austin/Desktop/rejector_xaltocan/downsampled_simulated"+ filenum +".vcf --silent --homozyg --homozyg-snp 50 --homozyg-window-missing 2 --homozyg-window-het 1 --homozyg-kb 500 --out /home/austin/Desktop/rejector_xaltocan/downsampled_simulated_ROH"+ filenum
            subprocess.call(rohcommand, shell=True)

            #read file back into python
            r_df = robjects.r('read.table(file = "/home/austin/Desktop/rejector_xaltocan/downsampled_simulated_ROH{0}.hom", header = T)'.format(filenum))
            pd_df = pd.DataFrame.from_dict({ key : np.asarray(r_df.rx2(key)) for key in r_df.names })

            #delete files
            removecommand = "rm /home/austin/Desktop/rejector_xaltocan/*simulated*" + filenum + ".{log,nosex,hom,hom.indiv,hom.summary,vcf}"
            subprocess.call(removecommand, shell=True, executable='/bin/bash')

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
    print("is this running?")
    #new_accepted_sims.append([candidate_parameters[1],candidate_parameters[0],candidate_parameters[2],candidate_parameters[4],candidate_parameters[3]])
    new_accepted_sims = [candidate_parameters[1],candidate_parameters[0],candidate_parameters[2],candidate_parameters[4],candidate_parameters[3]]
    #we are calculating the new weights "new_start_weights
    #find the density of the candidate parameters in multidimensional distribution
    #based on the previous iteration's accepted parameter sets
    #for sim in range(0,numsims):
    #    theta_sim = dtmvnorm(x=base.as_vector(candidate_logparameters),
    #                         mean = robjects.FloatVector(np.log(accepted_sims_gbl[sim])),
    #                         sigma = Br,
    #                         log = "TRUE")
    #    weights_denom.append(np.log(start_weights_gbl[sim])+np.array(theta_sim[0]))
    rpy2.robjects.numpy2ri.activate()#activate the numpy-r convertor
    Br_transform = np.linalg.multi_dot([D_constraints,Br,np.transpose(D_constraints)])
    for sim in range(0,numsims):
        lower_bounds_transform_sim = lower_bounds - (np.dot(D_constraints,np.log(accepted_sims_gbl[sim])))
        arg_sim = np.dot(D_constraints,(np.transpose(candidate_logparameters) - np.transpose(np.log(accepted_sims_gbl[sim]))))
        theta_sim = dtmvnorm(x=robjects.FloatVector(arg_sim),
                             sigma = Br_transform,
                             lower = robjects.FloatVector(lower_bounds_transform_sim),
                             upper = robjects.FloatVector(upper_bounds), # transformed upper bounds are the same as upper_bounds
                             log = "TRUE")
        print(theta_sim)
        print(start_weights_gbl[sim])
        weights_denom.append(np.log(start_weights_gbl[sim])+np.array(theta_sim)[0])
    print(weights_denom)
    rpy2.robjects.numpy2ri.deactivate()#activate the numpy-r convertor
    #print(weights_denom)
    #find largest value in weights denom matrix
    print("is this running??")
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
    print("is this running???")
    #log_p = dtmvnorm(x=base.as_vector(candidate_logparameters),
    #                    mean = robjects.FloatVector(prior_mean),
    #                    sigma = prior_sigma,
    #                    log = "TRUE")
    #print(log_p)
    rpy2.robjects.numpy2ri.activate()#activate the numpy-r convertor
    lower_bounds_transformed = lower_bounds - (np.dot(D_constraints,prior_mean))
    #print(lower_bounds_transformed)
    prior_sigma_transformed = np.linalg.multi_dot([D_constraints,prior_sigma,np.transpose(D_constraints)])
    #print(prior_sigma_transformed)
    #also had to transpose like in the previous arg call
    arg_transformed = np.dot(D_constraints,(np.transpose(candidate_logparameters) - prior_mean))
    #print(arg_transformed)
    log_p = dtmvnorm(x = robjects.FloatVector(arg_transformed),
                     sigma = prior_sigma_transformed,
                     lower = robjects.FloatVector(lower_bounds_transformed),
                     upper = robjects.FloatVector(upper_bounds), # transformed upper bounds are the same as upper_bounds
                     log = "TRUE")
    print(log_p)
    rpy2.robjects.numpy2ri.deactivate()#activate the numpy-r convertor
    #calculate the weight for the i-th new accepted parameter set
    new_start_weights = np.exp(log_p - log_q)
    #print(new_start_weight)
    print("IS THIS EVEN RUNNING????")
    #append new weight to new_start_weights
    #new_start_weights.append(new_start_weight[0])
    #print("%d accepted simulations so far." % (len(new_accepted_sims)))
    #print(new_accepted_sims)
    #new_listodists.append(distance)
    new_listodists = distance #trying to do these outputs as singles instead of lists like the first function
    print(new_accepted_sims)
    return(new_accepted_sims,new_start_weights,new_listodists)





# Run first iteration
print("Starting threshold: " + str(thresholds[0]))
start_time = time.time()

p = multiprocessing.Pool(ncores)
accepted_sims_gbl = []
start_weights_gbl = []
listodists_gbl = []
p.map(first_run,range(numsims))
for accepted_sims,start_weights,listodists in p.map(first_run,range(numsims)):
    accepted_sims_gbl.append(accepted_sims)
    start_weights_gbl.append(start_weights)
    listodists_gbl.append(listodists)
p.close()
p.join()

#define the covariance matrix of the accepted parameter combinations, sigma
sigmat = 2*np.cov(np.log(np.array(accepted_sims_gbl).T))
rpy2.robjects.numpy2ri.activate()
nr,nc = sigmat.shape
Br = robjects.r.matrix(sigmat, nrow=nr, ncol=nc)
robjects.r.assign("sigmat", Br)
print("sigmat matrix")
print(sigmat)
rpy2.robjects.numpy2ri.deactivate()

print("--- %s seconds ---" % (time.time() - start_time))


### ADD WRITE STATEMENT HERE FOR PARAMETER VALUES FROM THIS ITERATION





# Run additional iterations
for t in tqdm(range(1,len(thresholds))):
    print("New threshold: " + str(thresholds[t]))
    threshold = thresholds[t]

    start_time = time.time()

    p = multiprocessing.Pool(ncores)
    new_accepted_sims_gbl = []
    new_start_weights_gbl = []
    new_listodists_gbl = []
    p.map(subsequent_runs,range(numsims))
    for new_accepted_sims,new_start_weights,new_listodists in p.map(subsequent_runs,range(numsims)):
        new_accepted_sims_gbl.append(new_accepted_sims)
        new_start_weights_gbl.append(new_start_weights)
        new_listodists_gbl.append(new_listodists)
    p.close()
    p.join()
    print("--- %s seconds ---" % (time.time() - start_time))

    print("old sims")
    print(accepted_sims_gbl)
    print("new sims")
    print(new_accepted_sims_gbl)
    print("old weights")
    print(start_weights_gbl)

    #assign new accepted sims to accepted sims for next threshold loop iteration
    accepted_sims_gbl = new_accepted_sims_gbl

    print("did the re-assignment work?")
    print(accepted_sims_gbl)
    #once all new start weights are collected
    #need to turn them into probabilities
    #by dividing each one by the sum of the weights
    new_start_weights_gbl = [item for items in new_start_weights_gbl for item in items]
    sum_weights = np.sum(new_start_weights_gbl)
    print(sum_weights)

    print("old weights")
    print(start_weights_gbl)
    start_weights_gbl = new_start_weights_gbl/sum_weights
    print("new weights")
    print(start_weights_gbl)

    print("old sigmas")
    print(sigmat)
    #calculate sigmat for next threshold loop iteration
    sigmat = 2*np.cov(np.log(np.array(accepted_sims_gbl).T))
    print("new sigmas")
    print(sigmat)
    nr,nc = sigmat.shape
    rpy2.robjects.numpy2ri.activate()#activate the numpy-r convertor
    Br = robjects.r.matrix(sigmat, nrow=nr, ncol=nc)
    robjects.r.assign("sigmat", Br)
    rpy2.robjects.numpy2ri.deactivate()#deactivate the numpy-r convertor

#write accepted parameters to file
#np.savetxt(fname="/home/austin/Desktop/rejector_xaltocan/recombmap_chr22_accepted_parameters_t{}.txt".format(thresholds[t]),X=accepted_sims,delimiter="\t")
#write weights of accepted parameters to file
#np.savetxt(fname="/home/austin/Desktop/rejector_xaltocan/recombmap_chr22_weights_t{}.txt".format(thresholds[t]),X=start_weights,delimiter="\t")
