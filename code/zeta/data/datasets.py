"""Zeta datasets"""
import os
from itertools import compress
import mne 
import pickle

def open_dict(name):
    with open(name, 'rb') as handle:
        b = pickle.load(handle)
    return(b)

def get_subjects(data_folder,verbose=False):
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
    print(verbose)
    assert os.path.exists(data_folder), "folder doesn't exist"
    sub_fold= os.listdir(data_folder)
    if verbose: print("voici les subfolders") 
    if verbose: print(sub_fold)
    #TODO: NOT WORKING IF THERE IS A FILE
    #TODO: NEED TO EXTRACT ONLY FOLDERS
    names = os.listdir(data_folder+os.path.sep +sub_fold[0])
    names2 = os.listdir(data_folder+os.path.sep +sub_fold[1])
    
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
    if verbose: print("voici les patients des tests" + str(cong.keys()))
    return cong

def get_raw(data_folder, subject, dico=None, cropping=0,adjust_record_length=False, interpolate = False):
    """loads and returns the raws associated with a subject within a database

    Parameters
    ----------
    data_folder : str
        path of the folder containing the data
    dico : UNUSED #TODO: remove
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
                runs.append(_annotate_audio(mne.io.read_raw_fif(filepath), cropping = cropping, adjust_record_length=adjust_record_length, interpolate = interpolate))
                labels.append(file.split('_')[1])
                      
            
    runs_0=list(compress(runs,[x=='low' for x in labels]))
    runs_1=list(compress(runs,[x=='high' for x in labels]))
    
    raw_0=mne.concatenate_raws(runs_0)
    raw_1=mne.concatenate_raws(runs_1)
    return raw_0, raw_1
    
    
def _annotate_audio(RAW, cropping = 0, adjust_record_length=False, interpolate = False):
    
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
    patient = RAW.filenames[0].split("\\")[-1]
    annocheser2=RAW._annotations
    for i in range(int(float(len(RAW))/(RAW.info["sfreq"]*7))+1):
        annocheser2.append([i*7], [5], str("STIM_"+situation))
    RAW._annotations = annocheser2
    if not cropping == 0:
        print(float((len(RAW) / RAW.info["sfreq"])))
        RAW.crop(tmin=cropping,)
    if adjust_record_length == True:
        if (len(RAW) / RAW.info["sfreq"] )> 190:
            print("enregistrement plus long que 180s, cropping en cours")
            RAW.crop(tmin =0,tmax=180.2)
    if interpolate== True:
        ## TO DO à modifier ici 
        os.chdir("C:/Users/Zeta/gipsa_lab/collabo_louis/tinnituseeg/code/zeta/data/resources/")
        dicos= ["bad_chan_norena_high_metaData_stats", "bad_chan_norena_low_metaData_stats"]
        ## TODO que faire avec les enregistrements qui sont tellement mauvais que rejetes? Pour l'instant le cas est ignored
        
        if situation == dicos[0].split("_")[3]:
            d = open_dict(dicos[0])
            
            if not d.keys().__contains__(patient):
                print ("patient ou enregistrement trop mauvais pour en tirer qqchose de valable")
            else:
                to_interpo = d[patient]
                RAW.load_data()
                RAW.info["bads"]=to_interpo[0]
                RAW.interpolate_bads()
                #RAW.plot(scalings="auto")
        else:
            if situation == dicos[1].split("_")[3]:
                d = open_dict(dicos[1])
                if not d.keys().__contains__(patient):
                    print ("patient ou enregistrement trop mauvais pour en tirer qqchose de valable")
                else:
                    to_interpo = d[patient]
                    RAW.load_data()
                    RAW.info["bads"]=to_interpo[0]
                    RAW.interpolate_bads()
                    #RAW.plot(scalings="auto")
    return RAW


## Testing
if __name__ == '__main__':
    data_folder= "C:/Users/Zeta/gipsa_lab/test_traitement/2/raw_fif/"
    dico = get_subjects(data_folder)
    raw0, raw1 = get_raw(data_folder, dico.keys()[1], dico = dico, interpolate = True)

