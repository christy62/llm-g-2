
import numpy as np
import torch
import matplotlib.pyplot as plt
import json
import random
import pandas as pd
from datasets import Dataset
from datasets import load_dataset
import json
import re
import torch, gc
import os
import ast
plt.style.use('ggplot')
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"]= "expandable_segments:True"
# encoding pattern:
# layers*nstrips+ strip [0:383]
nstrips = 32
nlayers = 12
min_strip=0
max_strip=nstrips-1
min_layer=0
max_layer=nlayers-1
special_array_tokens =[]

# pattern=  r'gen_event\(slp=-?\d+\.\d+,icpt=-?\d+\.\d+(?:,rem=\[(\d+, )*\d*\])?(?:,fill=\[(\d+, )*\d*\])?\)(?:\+\[(\d+, )*\d*\])?$'


def is_valid_gen_event(input_str):
    pattern = r"""
    ^gen_event\(
    slp=\[(-?(?:0|[1-9]\d*)(?:\.\d+)?(?:,\s*-?(?:0|[1-9]\d*)(?:\.\d+)?)*)?\],
    icpt=\[(-?(?:0|[1-9]\d*)(?:\.\d+)?(?:,\s*-?(?:0|[1-9]\d*)(?:\.\d+)?)*)?\],
    add=\[((?:0|[1-9]\d*)(?:,\s*(?:0|[1-9]\d*))*)?\],
    rem=\[((?:0|[1-9]\d*)(?:,\s*(?:0|[1-9]\d*))*)?\],
    fill=\[((?:0|[1-9]\d*)(?:,\s*(?:0|[1-9]\d*))*)?\]
    \)$
    """.replace("\n", "").replace(" ", "")
    return bool(re.fullmatch(pattern, input_str))

# this was added on 03 Oct 2025 to check for the lengths of slp and icpt
# def is_valid_gen_event(input_str):
#     pattern = r"""
#     ^gen_event\(
#     slp=\[(-?(?:0|[1-9]\d*)(?:\.\d+)?(?:,\s*-?(?:0|[1-9]\d*)(?:\.\d+)?)*)?\],
#     icpt=\[(-?(?:0|[1-9]\d*)(?:\.\d+)?(?:,\s*-?(?:0|[1-9]\d*)(?:\.\d+)?)*)?\],
#     add=\[((?:0|[1-9]\d*)(?:,\s*(?:0|[1-9]\d*))*)?\],
#     rem=\[((?:0|[1-9]\d*)(?:,\s*(?:0|[1-9]\d*))*)?\],
#     fill=\[((?:0|[1-9]\d*)(?:,\s*(?:0|[1-9]\d*))*)?\]
#     \)$
#     """.replace("\n", "").replace(" ", "")

#     match = re.fullmatch(pattern, input_str)
#     if not match:
#         return False

#     # Extract the slp and icpt arrays
#     slp_str = match.group(1) or ""
#     icpt_str = match.group(2) or ""

#     # Count the number of elements in each array
#     slp_count = len(slp_str.split(',')) if slp_str else 0
#     icpt_count = len(icpt_str.split(',')) if icpt_str else 0

#     # Return True only if lengths match
#     return slp_count == icpt_count

# added on 05 Oct 2025
# def is_valid_gen_event(input_str):
#     # Pattern for a single gen_event
#     single_event_pattern = r"""
#     gen_event\(
#     slp=\[(-?(?:0|[1-9]\d*)(?:\.\d+)?(?:,\s*-?(?:0|[1-9]\d*)(?:\.\d+)?)*)?\],
#     icpt=\[(-?(?:0|[1-9]\d*)(?:\.\d+)?(?:,\s*-?(?:0|[1-9]\d*)(?:\.\d+)?)*)?\],
#     add=\[((?:0|[1-9]\d*)(?:,\s*(?:0|[1-9]\d*))*)?\],
#     rem=\[((?:0|[1-9]\d*)(?:,\s*(?:0|[1-9]\d*))*)?\],
#     fill=\[((?:0|[1-9]\d*)(?:,\s*(?:0|[1-9]\d*))*)?\]
#     \)
#     """.replace("\n", "").replace(" ", "")

#     # Pattern for the entire string: one or more gen_event() calls separated by '+'
#     full_pattern = rf"^{single_event_pattern}(\s*\+\s*{single_event_pattern})*$"

#     return bool(re.fullmatch(full_pattern, input_str))

def parse_gen_event_params(func_string):
    """
    Parse gen_event function string and extract parameter arrays
    Handles various formats including: slp=[-3.44],icpt=[29.46]
    """
    # Pattern to match the function parameters (more flexible)
    pattern = r'gen_event\(([^)]+)\)'
    match = re.search(pattern, func_string)

    if not match:
        raise ValueError("Could not find gen_event parameters")

    params_str = match.group(1)

    # Extract individual parameter arrays using more flexible regex
    param_patterns = {
        'slp': r'slp=\[([^\]]*)\]',
        'icpt': r'icpt=\[([^\]]*)\]',
        'add': r'add=\[([^\]]*)\]',
        'rem': r'rem=\[([^\]]*)\]',
        'fill': r'fill=\[([^\]]*)\]'
    }

    result = {}
    for param_name, pattern in param_patterns.items():
        match = re.search(pattern, params_str)
        if match:
            # Extract the array content and parse it
            array_content = match.group(1).strip()
            if array_content:  # If not empty
                try:
                    # Use ast.literal_eval for safe evaluation
                    result[param_name] = ast.literal_eval(f"[{array_content}]")
                except (ValueError, SyntaxError):
                    # Handle cases where parsing fails
                    # Split by commas and try to convert to appropriate types
                    items = []
                    for item in array_content.split(','):
                        item = item.strip()
                        try:
                            # Try to convert to float first, then int
                            items.append(float(item))
                        except ValueError:
                            try:
                                items.append(int(item))
                            except ValueError:
                                items.append(item)  # Keep as string if all else fails
                    result[param_name] = items
            else:
                result[param_name] = []
        else:
            result[param_name] = []  # Default empty array

    return result



def get_track_similarity(in_code, out_code):
    pt = parse_gen_event_params(in_code)
    po = parse_gen_event_params(out_code)
    if len(po['slp'])-len(po['icpt']):
        print("Warning: slope and icpt lengths do not match")
        print(in_code, out_code)

    if (len(pt['slp'])-len(po['slp'])) ==0:
        if len(pt['icpt'])-len(po['icpt'])!=0:
            print("Warning: icpt lengths do not match")
            print(in_code, out_code)
    # print(in_code,out_code)
    return len(pt['slp'])-len(po['slp'])

def get_similarity(in_array, out_array):
    """
    Calculate the similarity between two arrays.
    The similarity is defined as the number of common elements divided by the total number of unique elements in both arrays.
    """
    in_set = set(in_array)
    out_set = set(out_array)
    common_elements = in_set.intersection(out_set)
    # total_unique_elements = len(in_set.union(out_set))
    # print(len(common_elements), total_unique_elements)
    if len(in_array) == 0:
        return 0.0  # Avoid division by zero
    return len(common_elements) / len(in_array)



def get_dsimilarity(in_array, out_array):
    """
    The number of elements in out_array that are not in in_array, normalized by the number of unique elements in out_array.
    """
    # print(in_array, out_array)
    in_set = set((in_array))
    out_set = set((out_array))
  
    unique_out_elements = out_set.difference(in_set)
    if len(out_set) == 0:
        return 1.0  # Avoid division by zero
    return len(unique_out_elements) / len(out_set)

def get_layershit(slp, icpt):
    """
    Returns a list of layers that were hit based on the slope and intercept.
    """
    layers = []
    for layer in range(min_layer, max_layer + 1):
        strip = int(slp * layer + icpt)
        if ((strip < max_strip + 1) and (strip > min_strip - 1)):
            layers.append(layer)
    return layers


def eval_gen_event(func_string):# evaluate the gen_event function string
    return eval(func_string)

def gen_event(slp=[], icpt=[], add=[], rem=[], fill=[]):
  event = []
  if len(slp)!=len(icpt):
    event.append(-998)
    return event
  for layer in range(min_layer,max_layer+1):
    if layer in fill:
        for i in range(min_strip,max_strip+1):
            hit = layer*(max_strip+1)+i
            if hit not in rem and hit not in event:
                event.append(hit)
    else:
        for s, i in zip(slp, icpt):
            strip = int(s*layer + i)
            if ((strip<max_strip+1) and (strip > min_strip-1)):
                hit = layer*(max_strip+1)+strip
                if hit not in rem and hit not in event:
                    event.append(hit)
  for hit in add:
    if hit not in rem and hit not in event:
        event.append(hit)
  event.sort()
  return event


def fill_layers(layers):
    event =[]
    for layer in layers:
        for i in range(min_strip,max_strip+1):
            event.append(layer*(max_strip+1)+i)
    return event


# add a hit to the event
def add_hits(hits):
    event =[]
    for hit in hits:
        event.append(hit)
    return event

#remove a strip from the layer
def remove_hits(evt, hits):
    for hit in hits:
        if hit in evt:
            evt.remove(hit)
    return evt

#return a set a of random layers
def rand_layers(min_layer=0, max_layer=11):
    size = np.random.randint(1,10)
    layers =[]
    while len(layers)<size:
        l=np.random.randint(low=min_layer, high=max_layer+1)
        if l not in layers:
            layers.append(l) # add layer number to be removed
    return layers

def rand_hits_not_in_event(event, size=10):
    # size = np.random.randint(1,max_layer+1*max_strip+1)
    hits = []
    while len(hits)<size:
        layer = np.random.randint(low=min_layer, high=max_layer+1)
        strip = np.random.randint(low=min_strip, high=max_strip+1)
        hit = layer*(max_strip+1)+strip
        if hit not in hits and hit not in event:
            hits.append(hit)
    return hits

def rand_hits_in_event(event):
    hits =[]
    if(len(event)>15):
        return random.sample(event, np.random.randint(0, 10))
    if(len(event)>5 and len(event)<15):
        return random.sample(event, np.random.randint(0, 5))
    return hits

def plot_event(events):
    plt.figure(figsize=(10, 6))
    plt.xlabel('Strip')
    plt.ylabel('Layer')
    plt.title('Layer vs Strip')
    plt.grid(True)
    plt.xlim(0,nstrips)
    plt.ylim(0,nlayers)
    colors = ["blue","orange","red","green","black","gray"]
    markers = ["<",">","p","-","*","x"]
    # markers = ["x",">","p","-","*","x"]
    labels = ["true", "pred"]
    # labels = ["hits"]
    i =0
    for evt in events:
        layers = [x // nstrips for x in evt]
        strips = [x % nstrips for x in evt]
        plt.scatter(strips, layers, color=colors[i], marker=markers[i], label=labels[i])
        i += 1
    plt.legend()
    plt.show()

def generate_data(types, events_per_type, fileName):
    training_data =[]
    for i, type in enumerate(types):
        events = events_per_type[i]
        #0: pristine
        if type == 0:
            i = 0

            while i != events:
                slp_array =[]
                icpt_array =[]
                slp_array.append(round(random.uniform(-10, 10), 2))
                icpt_array.append(round(random.uniform(-5, 36), 2))
                code = f"gen_event(slp={slp_array},icpt={icpt_array},add=[],rem=[],fill=[])"
                event = eval(code)
                event.sort()
                if len(event) > 5:  # check if the event is not empty
                    #training_data.append((event, code))training_data.append((eval(code), code))
                    i += 1
                    if i==1:
                      print(code)
                      if is_valid_gen_event(code):
                        print("Pattern match verified")
                      else:
                        print("Pattern match failed")
            print(f"Generated {i} pristine events.")


        #1: pristine plus random hits
        if type == 1:
            i = 0
            while i != events:
                slp_array =[]
                icpt_array =[]
                slp_array.append(round(random.uniform(-10, 10), 2))
                icpt_array.append(round(random.uniform(-5, 36), 2))
                if len(gen_event(slp_array, icpt_array)) < 5: # all event that do not produce a track will have a slope of 0 and intercept of -1
                    slp_array =[]
                    icpt_array =[]
                nhits = random.randint(1, 5)
                rand_hits = random.sample(range(383), nhits)
                hits = rand_hits_not_in_event(gen_event(slp_array, icpt_array),nhits)
                code = f"gen_event(slp={slp_array},icpt={icpt_array},add={hits},rem=[],fill=[])"
                event = eval(code)
                event.sort()
                if (len(event)) > 5:
                    #training_data.append((event, code))training_data.append((eval(code), code))
                    i += 1
                    if i==1:
                      print(code)
                      if is_valid_gen_event(code):
                        print("Pattern match verified")
                      else:
                        print("Pattern match failed")
            print(f"Generated {i} pristine with add_hits events.")
        #2: pristine minus random hits
        if type == 2:
            i = 0

            while i != events:
                slp_array =[]
                icpt_array =[]
                slp_array.append(round(random.uniform(-10, 10), 2))
                icpt_array.append(round(random.uniform(-5, 36), 2))
                if len(gen_event(slp_array, icpt_array)) < 5: # all event that do not produce a track will have a slope of 0 and intercept of -1
                    slp_array =[]
                    icpt_array =[]
                hits = rand_hits_in_event(gen_event(slp_array, icpt_array))
                code = f"gen_event(slp={slp_array},icpt={icpt_array},add=[],rem={hits},fill=[])"
                event = eval(code)
                event.sort()
                if (len(event)) > 5:
                    #training_data.append((event, code))
                    training_data.append((eval(code), code))
                    i += 1
                    if i==1:
                      print(code)
                      if is_valid_gen_event(code):
                        print("Pattern match verified")
                      else:
                        print("Pattern match failed")
            print(f"Generated {i} pristine with rem_hits events.")

        #3: pristine with fill layers and random hits
        if type == 3:
            i = 0

            while i != events:
                slp_array =[]
                icpt_array =[]
                slp_array.append(round(random.uniform(-10, 10), 2))
                icpt_array.append(round(random.uniform(-5, 36), 2))
                if len(gen_event(slp_array, icpt_array)) == 0: # all event that do not produce a track will have a slope of 0 and intercept of -1
                    slp_array =[]
                    icpt_array =[]
                nfill = random.randint(1, 5)
                fill_l = random.sample(range(12), nfill)
                nhits = random.randint(1, 5)
                rand_hits = random.sample(range(383), nhits)
                hits = rand_hits_not_in_event(gen_event(slp_array, icpt_array, fill=fill_l),nhits)
                code = f"gen_event(slp={slp_array},icpt={icpt_array},add={hits},rem=[],fill={fill_l})"
                event = eval(code)
                event.sort()
                # event_prim = gen_event(slp, icpt)
                if (len(event)) > 5:
                    #training_data.append((event, code))
                    training_data.append((eval(code), code))
                    i += 1
                    if i==1:
                      print(code)
                      if is_valid_gen_event(code):
                        print("Pattern match verified")
                      else:
                        print("Pattern match failed")
            print(f"Generated {i} pristine plus random hits with fill layers events.")
        #4: pristine with fill layers rem and add
        if type == 4:
            i = 0
            while i != events:
                slp_array =[]
                icpt_array =[]
                slp_array.append(round(random.uniform(-10, 10), 2))
                icpt_array.append(round(random.uniform(-5, 36), 2))
                if len(gen_event(slp_array, icpt_array)) > 5:
                    nfill = random.randint(0, 2)
                    fill_l = random.sample(range(12), nfill)
                    fill_l.sort()

                    nhits = random.randint(0, 5)
                    if nhits==0:
                        add_hits=[]
                    else:
                        add_hits = rand_hits_not_in_event(gen_event(slp_array, icpt_array,fill=fill_l),nhits)
                        add_hits.sort()

                    rhits = random.randint(0, 1)
                    if rhits==0:
                        rem_hits=[]
                    else:
                        rem_hits = rand_hits_in_event(gen_event(slp_array, icpt_array, fill=fill_l))
                        rem_hits.sort()

                    code = f"gen_event(slp={slp_array},icpt={icpt_array},add={add_hits},rem={rem_hits},fill={fill_l})"

                    event = eval(code)
                    event.sort()

                    if (len(event)) > 5:
                        #training_data.append((event, code))
                        training_data.append((eval(code), code))
                        i += 1
                        if i==1:
                            print(code)
                            if is_valid_gen_event(code):
                                print("Pattern match verified")
                            else:
                                print("Pattern match failed")
            print(f"Generated {i} pristine plus random hits with fill layers, add and rem events.")

#48: pristine with fill layers rem and add with multiplicity effects
        if type == 48:
            i = 0
            while i != events:
                slp_array =[]
                icpt_array =[]
                slp_array.append(round(random.uniform(-10, 10), 2))
                icpt_array.append(round(random.uniform(-5, 36), 2))
                
                # layers not having any hits
                # n_skip_layers = random.randint(0,3)
                n_skip_layers = random.choices([0,1,2,3,4,5], weights=[0.20,0.20,0.20,0.20,0.10,0.10])[0] 
                skip_layers = random.sample(range(12), n_skip_layers)
                

                if len(gen_event(slp_array, icpt_array)) > 7: ## changes from 5 to 7
                    nfill = random.randint(0, 2)
                    fill_l = random.sample(range(12), nfill)
                    fill_l = [l for l in fill_l if l not in skip_layers]  # remove skip layers
                    fill_l.sort()

                    nhits = random.randint(0, 5)
                    if nhits==0:
                        add_hits=[]
                    else:
                        add_hits = rand_hits_not_in_event(gen_event(slp_array, icpt_array,fill=fill_l),nhits)
                        add_hits.sort()

                    rhits = random.randint(0, 1)
                    if rhits==0:
                        rem_hits=[]
                    else:
                        rem_hits = rand_hits_in_event(gen_event(slp_array, icpt_array, fill=fill_l))
                        rem_hits.sort()

                         
                    code = f"gen_event(slp={slp_array},icpt={icpt_array},add={add_hits},rem={rem_hits},fill={fill_l})"
                    event = eval(code)
                    event.sort()

                    # now add multiplicity effects
                    fill_layer_hits = gen_event(fill=fill_l)
                    mult_event =[]
                    for hit in event:
                        if hit not in fill_layer_hits:
                            mult = random.choices([1,2,3,4], weights=[0.4,0.25,0.25,0.1])[0] 
                            if mult==1: # no modification
                                mult_event.append(hit)
                            if mult==2: # add to left
                                if (hit-1)>0:
                                    mult_event.append(hit)
                                    mult_event.append(hit-1)
                                    add_hits.append(hit-1)
                            if mult==3: # add to right
                                if (hit+1)<383:
                                    mult_event.append(hit)
                                    mult_event.append(hit+1)
                                    add_hits.append(hit+1)
                            if mult==4: # add both sides
                                if (hit-1)>0 and (hit+1)<383:
                                    mult_event.append(hit)
                                    mult_event.append(hit-1)
                                    mult_event.append(hit+1)
                                    add_hits.append(hit-1)
                                    add_hits.append(hit+1)
                        else:
                            mult_event.append(hit)
                            
                    mult_event = list(set(mult_event)) # remove duplicates

                    skip_layer_hits = gen_event(fill=skip_layers)
                    
                    rem_hits02 = [h for h in mult_event if h in skip_layer_hits]
                    mult_event = [h for h in mult_event if h not in skip_layer_hits]
                    add_hits = [h for h in add_hits if h not in skip_layer_hits]
                    add_hits.sort()

                    
                    
                    rem_hits.extend(rem_hits02)
                    rem_hits.sort()
                    add_hits = [h for h in add_hits if h not in rem_hits]
                    add_hits.sort() 

                    mult_event.sort()

                    code = f"gen_event(slp={slp_array},icpt={icpt_array},add={add_hits},rem={rem_hits},fill={fill_l})" # modified add hits
                    event = mult_event
                    
                    # verify if the generated track is long enough
                    track_code = f"gen_event(slp={slp_array},icpt={icpt_array})"
                    track_event = eval(track_code)
                    if (len(track_event)) > 5 and len(eval(code))>5:
                        training_data.append((eval(code), code))
                        i += 1
                        if i==1:
                            print(code)
                            if is_valid_gen_event(code):
                                print("Pattern match verified")
                            else:
                                print("Pattern match failed")
            print(f"Generated {i} pristine plus random hits with fill layers, add and rem events.")

#49: pristine with fill layers rem and add with multiplicity effects but with modified first layer hits to prevent biased intercepts
        if type == 49:
            i = 0
            while i != events:
                slp_array =[]
                icpt_array =[]
                slp_array.append(round(random.uniform(-10, 10), 2))
                icpt_array.append(round(random.uniform(-5, 36), 2))
                
                # layers not having any hits
                # n_skip_layers = random.randint(0,3)
                n_skip_layers = random.choices([0,1,2,3,4,5], weights=[0.20,0.20,0.20,0.20,0.10,0.10])[0] 
                skip_layers = random.sample(range(12), n_skip_layers)
                

                if len(gen_event(slp_array, icpt_array)) > 7: ## changes from 5 to 7
                    nfill = random.randint(0, 2)
                    fill_l = random.sample(range(12), nfill)
                    fill_l = [l for l in fill_l if l not in skip_layers]  # remove skip layers
                    fill_l.sort()

                    nhits = random.randint(0, 5)
                    if nhits==0:
                        add_hits=[]
                    else:
                        add_hits = rand_hits_not_in_event(gen_event(slp_array, icpt_array,fill=fill_l),nhits)
                        add_hits.sort()

                    rhits = random.randint(0, 1)
                    if rhits==0:
                        rem_hits=[]
                    else:
                        rem_hits = rand_hits_in_event(gen_event(slp_array, icpt_array, fill=fill_l))
                        rem_hits.sort()

                         
                    code = f"gen_event(slp={slp_array},icpt={icpt_array},add={add_hits},rem={rem_hits},fill={fill_l})"
                    event = eval(code)
                    event.sort()

                    # now add strip hit multiplicity effects
                    fill_layer_hits = gen_event(fill=fill_l)
                    mult_event =[]
                    for hit in event:
                        if hit not in fill_layer_hits:
                            mult = random.choices([1,2,3,4], weights=[0.4,0.25,0.25,0.1])[0] 
                            if mult==1: # no modification
                                mult_event.append(hit)
                            if mult==2: # add to left
                                if (hit-1)>0:
                                    mult_event.append(hit)
                                    mult_event.append(hit-1)
                                    add_hits.append(hit-1)
                            if mult==3: # add to right
                                if (hit+1)<383:
                                    mult_event.append(hit)
                                    mult_event.append(hit+1)
                                    add_hits.append(hit+1)
                            if mult==4: # add both sides
                                if (hit-1)>0 and (hit+1)<383:
                                    mult_event.append(hit)
                                    mult_event.append(hit-1)
                                    mult_event.append(hit+1)
                                    add_hits.append(hit-1)
                                    add_hits.append(hit+1)
                        else:
                            mult_event.append(hit)
                            
                    mult_event = list(set(mult_event)) # remove duplicates

                    skip_layer_hits = gen_event(fill=skip_layers)
                    
                    rem_hits02 = [h for h in mult_event if h in skip_layer_hits]
                    mult_event = [h for h in mult_event if h not in skip_layer_hits]
                    add_hits = [h for h in add_hits if h not in skip_layer_hits]
                    add_hits.sort()

                    
                    
                    rem_hits.extend(rem_hits02)
                    rem_hits.sort()
                    add_hits = [h for h in add_hits if h not in rem_hits]
                    add_hits.sort() 

                    mult_event.sort()

                    code = f"gen_event(slp={slp_array},icpt={icpt_array},add={add_hits},rem={rem_hits},fill={fill_l})" # modified add hits
                    event = mult_event
                    
                    # verify if the generated track is long enough
                    track_code = f"gen_event(slp={slp_array},icpt={icpt_array})"
                    track_event = eval(track_code)

                    #check the hit in the bottom layer and remove the hits closer to intercept
                    icpt = icpt_array[0]
                    h = [h for h in event if abs(h-icpt)<2]
                    if len(h)>0:
                        rem_hits.append(h[0])
                        rem_hits.sort()
                        disp=random.randint(3, 8) # add a random displacement hit

                        if random.randint(0, 1)==0: # add
                            add_h = h[0]+disp
                            if add_h<383 and add_h not in rem_hits:
                                add_hits.append(add_h)
                            else:
                                add_h = h[0]-disp
                                if add_h>0 and add_h not in rem_hits:
                                    add_hits.append(add_h)
                        else: # subtract
                            add_h = h[0]-disp
                            if add_h>0 and add_h not in rem_hits:
                                add_hits.append(add_h)
                            else:
                                add_h = h[0]+disp
                                if add_h<383 and add_h not in rem_hits:
                                    add_hits.append(add_h)
                    
                    add_hits = [h for h in add_hits if h not in rem_hits]
                    add_hits.sort()
                    code = f"gen_event(slp={slp_array},icpt={icpt_array},add={add_hits},rem={rem_hits},fill={fill_l})" # modified add hits


                    if (len(track_event)) > 5 and len(eval(code))>5:
                        training_data.append((eval(code), code))
                        i += 1
                        if i==1:
                            print(code)
                            if is_valid_gen_event(code):
                                print("Pattern match verified")
                            else:
                                print("Pattern match failed")
            print(f"Generated {i} pristine plus random hits with fill layers, add and rem events.")



        #4: random noise only
        if type == 40:
            i = 0
            while i != events:
                slp_array =[]
                icpt_array =[]

                nfill = random.randint(0, 2)
                fill_l = random.sample(range(12), nfill)
                fill_l.sort()

                nrl = random.randint(6, 12) # number of layers to choose
                n_random_layers = random.sample(range(0,12),nrl)  # choose random layers

                add_hits=[]
                for l in n_random_layers:
                   strip = random.randint(0, 31)
                   if (strip<max_strip+1) and (strip > min_strip-1):
                       if l not in fill_l:
                            hit = l*(max_strip+1)+strip
                            add_hits.append(hit)

                add_hits.sort()
                nhits = random.randint(0, 5)
                if nhits!=0:
                    add_hits = rand_hits_not_in_event(gen_event(slp_array, icpt_array, add=add_hits, fill=fill_l),nhits)
                    add_hits.sort()

                rhits = random.randint(0, 1)
                if rhits==0:
                    rem_hits=[]
                else:
                    rem_hits = rand_hits_in_event(gen_event(slp_array, icpt_array, fill=fill_l))
                    rem_hits.sort()

                code = f"gen_event(slp={slp_array},icpt={icpt_array},add={add_hits},rem={rem_hits},fill={fill_l})"

                event = eval(code)
                event.sort()

                if (len(add_hits)) > 5:
                    #training_data.append((event, code))
                    training_data.append((eval(code), code))
                    i += 1
                    if i==1:
                        print(code)
                        if is_valid_gen_event(code):
                            print("Pattern match verified")
                        else:
                            print("Pattern match failed")
            print(f"Generated {i} random noise hits.")

        #5: multievent with noise
        if type == 5:
            i = 0

            while i != events:
                slp_array =[]
                icpt_array =[]
                n_events= random.randint(2, 4)
                n=0
                while n<n_events:
                    slp = round(random.uniform(-10, 10), 2)
                    icpt = round(random.uniform(-5, 36), 2)
                    if len(gen_event([slp], [icpt])) > 5:  # ensure that the event is not empty
                        slp_array.append(slp)
                        icpt_array.append(icpt)
                        n = n+1


                # if len(gen_event(slp_array, icpt_array)) < 8: # all event that do not produce a track will have a slope of 0 and intercept of -1
                #     slp_array =[]
                #     icpt_array =[]

                nfill = random.randint(0, 2)
                fill_l = random.sample(range(12), nfill)
                fill_l.sort()

                nhits = random.randint(0, 5)
                if nhits==0:
                    add_hits=[]
                else:
                    add_hits = rand_hits_not_in_event(gen_event(slp_array, icpt_array,fill=fill_l),nhits)
                    add_hits.sort()

                rhits = random.randint(0, 1)
                if rhits==0:
                    rem_hits=[]
                else:
                    rem_hits = rand_hits_in_event(gen_event(slp_array, icpt_array, fill=fill_l))
                    rem_hits.sort()

                code = f"gen_event(slp={slp_array},icpt={icpt_array},add={add_hits},rem={rem_hits},fill={fill_l})"

                event = eval(code)
                event.sort()

                if (len(event)) > 5:
                    #training_data.append((event, code))
                    training_data.append((eval(code), code))
                    if (len(event)) >150:
                      print(f"high hit event {code} {len(event)}")
                    i += 1
                    if i==1:
                      print(code)
                      if is_valid_gen_event(code):
                        print("Pattern match verified")
                      else:
                        print("Pattern match failed")

            print(f"Generated {i} multievent with fill layers, add and rem events.")

# clean multievent
        if type == 51:
            i = 0

            while i != events:
                slp_array =[]
                icpt_array =[]
                n_events= random.randint(2, 4)
                n=0
                while n<n_events:
                    slp = round(random.uniform(-10, 10), 2)
                    icpt = round(random.uniform(-5, 36), 2)
                    if len(gen_event([slp], [icpt])) > 5:  # ensure that the event is not empty
                        slp_array.append(slp)
                        icpt_array.append(icpt)
                        n = n+1


                # nfill = random.randint(1, 5)
                # fill_l = random.sample(range(12), nfill)

                # nhits = random.randint(1, 5)
                # add_hits = rand_hits_not_in_event(gen_event(slp_array, icpt_array,fill=fill_l),nhits)
                # rem_hits = rand_hits_in_event(gen_event(slp_array, icpt_array, fill=fill_l))

                code = f"gen_event(slp={slp_array},icpt={icpt_array},add=[],rem=[],fill=[])"

                event = eval(code)
                event.sort()

                if (len(event)) > 5:
                    #training_data.append((event, code))
                    training_data.append((eval(code), code))
                    if (len(event)) >150:
                      print(f"high hit event {code} {len(event)}")
                    i += 1
                    if i==1:
                      print(code)
                      if is_valid_gen_event(code):
                        print("Pattern match verified")
                      else:
                        print("Pattern match failed")

            print(f"Generated {i} pristine multievents...")

# split express
        if type == 50:
            i = 0
            while i != events:
                slp_array =[]
                icpt_array =[]
                n_tracks= random.randint(1, 4)
                n=0
                while n<n_tracks:
                    slp = round(random.uniform(-10, 10), 2)
                    icpt = round(random.uniform(-5, 36), 2)
                    if len(gen_event([slp], [icpt])) >= 5:  # ensure that the event is not empty
                        slp_array.append(slp)
                        icpt_array.append(icpt)
                        n = n+1


                # if len(gen_event(slp_array, icpt_array)) < 8: # all event that do not produce a track will have a slope of 0 and intercept of -1
                #     slp_array =[]
                #     icpt_array =[]

                nfill = random.randint(1, 5)
                fill_l = random.sample(range(12), nfill)

                nhits = random.randint(1, 5)
                add_hits = rand_hits_not_in_event(gen_event(slp_array, icpt_array,fill=fill_l),nhits)
                rem_hits = rand_hits_in_event(gen_event(slp_array, icpt_array, fill=fill_l))
                # print(slp_array,icpt_array)
                cc=[]
                for ss,ii in zip(slp_array,icpt_array):
                    cc.append(f"gen_event(slp=[{ss}],icpt=[{ii}],add=[],rem=[],fill=[])")
                cc.append(f"gen_event(slp=[],icpt=[],add={add_hits},rem={rem_hits},fill={fill_l})")

                # code = f"gen_event(slp={slp_array},icpt={icpt_array},add={add_hits},rem={rem_hits},fill={fill_l})"
                code = result = " + ".join(cc)
                event = eval(code)
                event.sort()

                if (len(event)) > 5:
                    #training_data.append((event, code))
                    training_data.append((eval(code), code))
                    i += 1
                    if i==1:
                      print(code)
                      if is_valid_gen_event(code):
                        print("Pattern match verified")
                      else:
                        print("Pattern match failed")

            print(f"Generated {i} multievent with fill layers, add and rem events.")


        #6: pristine multievent (this was not it training! added later for analysis)
        if type == 6:
            i = 0

            while i != events:
                slp_array =[]
                icpt_array =[]
                for j in range(random.randint(1, 5)):
                    slp = round(random.uniform(-10, 10), 2)
                    icpt = round(random.uniform(-5, 36), 2)
                    if len(gen_event([slp], [icpt])) >= 5:  # ensure that the event is not empty
                        slp_array.append(slp)
                        icpt_array.append(icpt)


                if len(gen_event(slp_array, icpt_array)) < 8: # all event that do not produce a track will have a slope of 0 and intercept of -1
                    slp_array =[]
                    icpt_array =[]

                # nfill = random.randint(1, 5)
                # fill_l = random.sample(range(12), nfill)

                # nhits = random.randint(1, 5)
                # add_hits = rand_hits_not_in_event(gen_event(slp_array, icpt_array,fill=fill_l),nhits)
                # rem_hits = rand_hits_in_event(gen_event(slp_array, icpt_array, fill=fill_l))
                add_hits=[]
                rem_hits =[]
                fill_l=[]

                code = f"gen_event(slp={slp_array},icpt={icpt_array},add={add_hits},rem={rem_hits},fill={fill_l})"

                event = eval(code)
                event.sort()

                if (len(event)) > 5:
                    #training_data.append((event, code))
                    training_data.append((eval(code), code))
                    i += 1
                    if i==1:
                      print(code)
                      if is_valid_gen_event(code):
                        print("Pattern match verified")
                      else:
                        print("Pattern match failed")

            print(f"Generated {i} pristine multievent.")

        #7: single track with atleast 10 hits with fill layers rem and add (this was not it training! added later for analysis)
        if type == 7:
            i = 0
            while i != events:
                slp_array =[]
                icpt_array =[]
                slp_array.append(round(random.uniform(-10, 10), 2))
                icpt_array.append(round(random.uniform(-5, 36), 2))
                while len(gen_event(slp_array, icpt_array)) < 9:
                    slp_array =[]
                    icpt_array =[]
                    slp_array.append(round(random.uniform(-10, 10), 2))
                    icpt_array.append(round(random.uniform(-5, 36), 2))
                # if len(gen_event(slp_array, icpt_array)) < 5: # all event that do not produce a track will have a slope of 0 and intercept of -1
                #     slp_array =[]
                #     icpt_array =[]

                nfill = random.randint(0, 5)
                fill_l = random.sample(range(-1,13), nfill) # -1 will make it empty

                nhits_add = random.randint(0, 10)
                # nhihts_rem = random.randint(0, 10)
                add_hits = rand_hits_not_in_event(gen_event(slp_array, icpt_array,fill=fill_l),nhits_add)
                rem_hits = rand_hits_in_event(gen_event(slp_array, icpt_array, fill=fill_l))

                code = f"gen_event(slp={slp_array},icpt={icpt_array},add={add_hits},rem={rem_hits},fill={fill_l})"

                event = eval(code)
                event.sort()

                if (len(event)) > 5:
                    #training_data.append((event, code))
                    training_data.append((eval(code), code))
                    i += 1
                    if i==1:
                      print(code)
                      if is_valid_gen_event(code):
                        print("Pattern match verified")
                      else:
                        print("Pattern match failed")
            print(f"Generated {i} pristine with atleast 5 hits plus random hits with fill layers, add and rem events.")


        #8: single track with atleast 5 hits with no fill layers rem and add (this was not it training! added later for analysis)
        if type == 8:
            i = 0
            nmin=11
            while i != events:
                slp_array =[]
                icpt_array =[]
                slp_array.append(round(random.uniform(-10, 10), 2))
                icpt_array.append(round(random.uniform(-5, 36), 2))
                while len(gen_event(slp_array, icpt_array)) < nmin:
                    slp_array =[]
                    icpt_array =[]
                    slp_array.append(round(random.uniform(-10, 10), 2))
                    icpt_array.append(round(random.uniform(-5, 36), 2))
                # if len(gen_event(slp_array, icpt_array)) < 5: # all event that do not produce a track will have a slope of 0 and intercept of -1
                #     slp_array =[]
                #     icpt_array =[]

                # nfill = random.randint(1, 5)
                # fill_l = random.sample(range(-1,13), nfill) # -1 will make it empty
                fill_l=[]
                nhits = random.randint(1, 5)
                add_hits = rand_hits_not_in_event(gen_event(slp_array, icpt_array,fill=fill_l),nhits)
                rem_hits = rand_hits_in_event(gen_event(slp_array, icpt_array, fill=fill_l))

                code = f"gen_event(slp={slp_array},icpt={icpt_array},add={add_hits},rem={rem_hits},fill={fill_l})"

                event = eval(code)
                event.sort()

                if (len(event)) > 5:

                    #training_data.append((event, code))
                    training_data.append((eval(code), code))

                    i += 1
                    if i==1:
                      print(code)
                      if is_valid_gen_event(code):
                        print("Pattern match verified")
                      else:
                        print("Pattern match failed")
            print(f"Generated {i} pristine with atleast {nmin} hits plus random hits with add and rem events and no fills.")



    training_data_json = []
    for event, code in training_data:
        training_data_json.append({"input": str(event), "target": code})
    with open(f"{fileName}.json", "w") as f:
        json.dump(training_data_json, f, indent=2)
        # return training_data_json
    print(f"{len(training_data_json)} events written to {fileName}")

def get_code_length(c):
    p=parse_gen_event_params(c)
    return len(p['slp'])+ len(p['slp'])+ len(p['rem'])+ + len(p['add']) + len(p['fill'])

def get_compression_ratio_exp(row):
    c=get_code_length(row['input_code'])
    a=len(row['input_array'])
    return 100*(a-c)/a

def get_compression_ratio_pred(row):
    c=get_code_length(row['pred_code'])
    a=len(row['input_array'])
    return 100*(a-c)/a

print("function definitions done...")

#%%%
