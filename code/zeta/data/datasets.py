"""Zeta datasets"""
import os
import numpy as np
from itertools import compress
import mne 

def get_subjects(data_folder):
	"""loads the list of data from a given folder

    Parameters
    ----------
    data_folder : str
        path of the folder containing the data

    Returns
    -------
    out : dict
        dictionary with mappings from session code to session metadata
    """
        if not os.path.exists(data_folder):
            print "error the folder path to access the data is not correct, check if " 
            + data_folder + " is a correct path on your computer"
            return
        else:
            sub_fold= os.listdir(data_folder)
            print "voici les subfolders"
            print sub_fold
            os.chdir(data_folder+sub_fold[0])
            names = os.listdir(data_folder+sub_fold[0])
            
            os.chdir(data_folder+sub_fold[1])
            names2 = os.listdir(data_folder+sub_fold[1])
            
            pat=[]
            pat2=[]
            for name in names:
                #print name.split('_')[0]
                pat.append(name.split('_')[0])
            for name in names2:
                #print name.split('_')[0]
                pat2.append(name.split('_')[0])
            #print pat
            cong={}
            for name in names2:
                    if pat.__contains__(name.split('_')[0]):
                        if cong.keys().__contains__(name.split('_')[0]):
                            cong[name.split('_')[0]].append(name)
                        else:
                            cong[name.split('_')[0]]=[name]
                            
                    
            for name in names:
                    if pat2.__contains__(name.split('_')[0]):
                        cong[name.split('_')[0]].append(name)
            print "voici les patients des tests " + str(cong.keys())
            return cong

def get_raw(data_folder, dico, subject):
    """loads and returns the raws associated with a subject within a database

    Parameters
    ----------
    data_folder : str
        path of the folder containing the data
    dico: 
        dictionary with mappings from session code to session metadata
    subject:
        subject ID from whose the raws will be returned

    Returns
    -------
    out : raw_0, raw_1
        concatenated files of the whole 0 and 1 setups, where all the 0 setup files
        have been concatenated together and all the 1 setup files have been 
        concataned together
    """
    runs=[]
    labels=[]
    for root, dirs, files in os.walk(data_folder):
        for file in files:
            if file.startswith(subject):
                filepath=os.path.join(root,file)
                # loads the annotated data
                runs.append(_annotate_audio(mne.io.read_raw_fif(filepath), cropping = 2, adjust_record_length=True))
                labels.append(file.split('_')[1])
                      
            
    runs_0=list(compress(runs,[x=='low' for x in labels]))
    runs_1=list(compress(runs,[x=='high' for x in labels]))
    
    raw_0=mne.concatenate_raws(runs_0)
    raw_1=mne.concatenate_raws(runs_1)
    return raw_0, raw_1
    
    
def _annotate_audio(RAW, cropping = 0, adjust_record_length=False):
    
    """ 
    annotate a raw file (totally uncleaned and uncropped) with stimulation events
    gives the option to crop the first 2 seconds of the data and to automatically crop
    the excedent data if this data is found to be twice as long as foreseen
    Parameters
    ----------
    RAW : mne.raw object
        recording to annotate
    cropping: int (optional)
        default 0: nothing done, if set to an int value, crops the beggining
        of the recording with the length of seconds given as an int
    adjust_record_length: boolean (optional)
        if set to True, detects if the recording is anormally long and crops the 
        exceedent data

    Returns
    -------
    out : RAW
        annotated and cropped raw file
    
    """
    situation = RAW.filenames[0].split("\\")[-1].split("_")[1]
    annocheser2=RAW.annotations
    for i in range(int(float(len(RAW))/(RAW.info["sfreq"]*7))+1):

                        annocheser2.append([i*7], [5], str("STIM_"+situation))

    RAW.annotations = annocheser2
    if not cropping == 0:
        print float((len(RAW) / RAW.info["sfreq"]))
        RAW.crop(tmin=cropping,)
    if (len(RAW) / RAW.info["sfreq"] )> 190:
        print "enregistrement plus long que 180s, cropping en cours"
        RAW.crop(tmin =0,tmax=180.2)
    return RAW


## Testing
data_folder= "C:/Users/Zeta/gipsa_lab/test_traitement/2/raw_clean_32/"
dico = get_subjects(data_folder)
raw0, raw1 = get_raw(data_folder, dico, dico.keys()[0])

