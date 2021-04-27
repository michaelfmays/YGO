import json, os, shutil, requests, textwrap
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from IPython.display import clear_output

def ImageDl(url):
    attempts = 0
    while attempts < 5: # Retry 5 times
        try:
            filename = url.split('/')[-1]
            r = requests.get(url, headers=headers,
                             stream=True, timeout=5)
            if r.status_code == 200:
                with open(os.path.join(path, filename), 'wb') as f:
                    r.raw.decode_content = True
                    shutil.copyfileobj(r.raw, f)
            print(filename)
            break
        except Exception as e:
            attempts+=1
            print(e)

def progress_update(i, L):
  # Progress update
    if i%100==0 and i!=(L-1):
      clear_output(wait=True)
      print(str((i*100)//L), "%", sep="")
    elif i==(L-1):
      clear_output(wait=True)
      print("Done!")
      return True

def update_filenames(current_dir, card):
  name1 = os.path.join(current_dir, "unsorted_cards",
                'Images', str(card)+'.jpg')
  ext = os.path.splitext(name1)[1]
  name2 = os.path.join(current_dir, "unsorted_cards",
                'Cropped', str(card)+"_cropped"+ext)
  return name1, name2

def crop_image(img, crop_area, new_filename):
    cropped_image = img.crop(crop_area)
    cropped_image.save(new_filename)

def make_splits(dat, split_on,
                train_prop, test_prop,
                rand_seed=453453):
  #              0.2 test
  # 0.75 * 0.8 = 0.6 train
  # 0.25 * 0.8 = 0.2 validation

  def calc_splits(te_fin, va_fin):
    return round(va_fin/(1-te_fin), 3)
  
  prop2 = calc_splits(test_prop, 1-(train_prop + test_prop))

  # Split data into (training + validation) and test
  X_train, X_test, y_train, y_test  = train_test_split(dat["id"],
                                                        dat[split_on],
                                  stratify=dat[split_on],
                                  test_size=test_prop,
                                  random_state=rand_seed)

  # Further split (training + validation) into (training) and (validation)
  X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                    y_train,
                                  stratify=dat[split_on].iloc[X_train.index],
                                  test_size=prop2,
                                  random_state=rand_seed+1) 
  
  return X_train, X_val

def move_card_images(dest_fldr, split_on="archetype",
                     train_prop=0.6, test_prop=0.2,
                     check_vector=None, cropped=True,
                     rand_seed=453453):
  """
  This function assumes:
    - Unsorted images exist in ./unsorted_cards/x/ for x in ["Cropped", "Images"]
    - You want images sorted into folders within ./sorted_cards/dest_fldr/
    - Pickled data frames for df (data_frame.pkl) and df_large (df_large.pkl)
         exist in ./data/
    - check_vector is an iterable 1D object with *desired archetype names*
  """

  currentDir = os.getcwd()

  # Input validation for split_on
  if check_vector is not None and split_on!="archetype":
    raise ValueError("Input check_vector should only be used with " +
                     "split_on = \'archetype\'.")

  if split_on not in ["archetype", "short_type", "type"]:
    raise ValueError("Must split_on one of: archetype, short_type, type")

  # Select correct source folder based on card type.
  # Also assign file suffix while we're at it.
  if cropped:
    source_fldr = os.path.join("unsorted_cards", "Cropped")
    suffix = "_cropped"
  else:
    source_fldr = os.path.join("unsorted_cards", "Images")
    suffix = ""

  # Check that source folder exists, raise error if not.
  # Expects source folder to exist WITHIN ./unsorted_cards/.
  if not os.path.exists(os.path.join(currentDir, source_fldr)):
    raise OSError("Source folder /" + str(source_fldr) +
                  " does not exist in current directory " +
                  str(currentDir))
    
  if split_on == "type":
    split_on = "general_type"
    
  # Pull correct data frame based on what type of split is being used
  if split_on=="archetype":
    d = pd.read_pickle(os.path.join(currentDir, "data", "df_large.pkl"))
    var_of_interest = d[split_on]
  elif split_on=="short_type" or split_on=="general_type":
    d = pd.read_pickle(os.path.join(currentDir, "data", "data_frame.pkl"))
    var_of_interest = d[split_on]

  X_train, X_val = make_splits(dat=d,
                               train_prop=train_prop,
                               test_prop=test_prop,
                               split_on=split_on,
                               rand_seed=rand_seed)

  # Prepend /sorted_cards/ to destination folder
  if "sorted_cards" not in dest_fldr:
    dest_fldr = os.path.join("sorted_cards", dest_fldr)

  # Check that destination folder exists, create it if not.
  if not os.path.exists(os.path.join(currentDir, dest_fldr)):
    os.makedirs(os.path.join(currentDir, dest_fldr))
  
  # Create train/test/valid in dest_fldr if they don't exist.
  for fldr in ["train", "test", "valid"]:
    if not os.path.exists(os.path.join(currentDir, dest_fldr, fldr)):
      os.makedirs(os.path.join(currentDir, dest_fldr, fldr))

  for i in range(len(d["id"])):
    progress_update(i, len(d["id"]))
    
    # If check_vector is given, skip the rest of the loop
    #    if the card's archetype is not in check_vector.
    if check_vector is not None and split_on=="archetype":
      if var_of_interest[i] not in check_vector:
        continue
    
    # Pick which dest_fldr sub-folder the image belongs in
    if i in X_train.index:
      grp = "train"
    elif i in X_val.index:
      grp = "valid"
    else:
      grp = "test"
    
    # Define filename and destination folder name
    file_name = str(d["id"][i])+suffix+".jpg"
    folder_name = os.path.join(currentDir, dest_fldr, grp,
                                      var_of_interest[i].replace("/", " "))
    
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
        shutil.copy(os.path.join(currentDir, source_fldr, file_name),
                    folder_name)
    elif not os.path.exists(os.path.join(folder_name, file_name)):
        shutil.copy(os.path.join(currentDir, source_fldr, file_name),
                    folder_name)