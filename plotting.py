import numpy as np
import matplotlib.pyplot as plt
#from decimal import *

ms_color = [0.12156863, 0.46666667, 0.70588235, 1]
hc_color = [1., 0.49803922, 0.05490196, 1]

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# set serif font
plt.rc('font', family='serif')

def get_labels_dict(path):
    import xmltodict
    with open(path) as f:
        labels_xml = xmltodict.parse(f.read())['atlas']['data']['label']

    labels_dict = {}
    for row in labels_xml:
        labels_dict[int(row['index'])] = row['name']
    return labels_dict

def heatmap_per_region(hm, atlas, positive=True, size_normalize=False, signed=False):
    # get heatmap mean per region
    # use only positive values
    signed_hm = np.copy(hm)
    if signed:
        if positive:
            signed_hm[signed_hm<0] = 0
        else:
            signed_hm[signed_hm>0] = 0
        
    regional_hm = {}
    for lbl_idx in np.unique(atlas):
        # skip outside area
        if lbl_idx != 0:
            atlas_lbl = atlas.copy()
            # get region mask for each label
            atlas_lbl[lbl_idx!=atlas] = 0
            atlas_lbl[lbl_idx==atlas] = 1
            # multiply region mask with heatmap
            region_intensity = np.mean(atlas_lbl * np.squeeze(signed_hm))
            if size_normalize:
                region_size = np.sum(atlas_lbl).item()
                region_intensity /= region_size
            regional_hm[lbl_idx] = region_intensity
    return regional_hm

def aggregate_regions(regional_hm, all_areas):
    # aggregate atlas regions to previously defined areas
    area_hm = {}
    for name, (min_idx, max_idx) in all_areas.items():
        regions_fit = []
        for key in regional_hm.keys():
            if key in range(min_idx, max_idx+1):
                regions_fit.append(regional_hm[key])
        region_mean = np.mean(regions_fit)
        area_hm[name] = region_mean
    return area_hm

def get_area_relevance(heatmaps, atlas, area_dict, positive=True, size_normalize=True):
    keys = []
    values = []
    for hm in heatmaps:
        regional_hm = heatmap_per_region(hm, atlas, positive=positive, size_normalize=size_normalize)
        area_hm = aggregate_regions(regional_hm, area_dict)
        # sort values
        area_hm_sorted = sorted(area_hm.items(), key=lambda kv: kv[1])
        keys_sorted = [row[0] for row in area_hm_sorted]
        values_sorted = [row[1] for row in area_hm_sorted]
        
        keys.append(keys_sorted)
        values.append(values_sorted)
    return keys, values

def plot_key_value_pairs(keys, values, title, loc="center left"):
    plt.figure(figsize=(10, 6))
    plt.plot(keys[0], values[0], 'o', color=ms_color, label="MS")
    plt.plot(keys[1], values[1], 'o', color=hc_color, label="HC")
    plt.xticks(rotation='vertical')
    plt.legend(loc=loc)
    plt.title(title)
    plt.show()

# Modified areas from Visualizing evidence for AD paper by
# Boehle et al. Based on Neuromorphometrics atlas from SPM12
# Name: (min, max)
gm_areas= {
    "Accumbens": (23, 30),
    "Amygdala": (31, 32),
    "Brain Stem": (35, 35),
    "Caudate": (36, 37),
    "Cerebellum": (38, 41),
    "Cerebral White Matter": (44, 45),
    "CSF" : (46, 46),
    "Hippocampus": (47, 48),
    "Parahippocampal gyrus": (170, 171),
    "Pallidum": (55, 56),
    "Putamen": (57, 58),
    "Thalamus": (59, 60),
    "Diencephalon": (61, 62),
    "ACG": (100, 101),
    "Anterior Insula": (102, 103),
    "Posterior Insula": (172, 173),
    "AOG": (104, 105),
    "AAG": (106, 107),
    "Cuneus": (114, 115),
    "Central operculum": (112, 113),
    "Frontal operculum": (118, 119),
    "Frontal pole": (120, 121),
    "Fusiform gyrus": (122, 123),
    "Temporal pole": (202, 203),
    "TrIFG": (204, 205),
    "TTG": (206, 207),
    "Ent": (116, 117),
    "Parietal operculum": (174, 175),
    "SPL": (198, 199),
}

# Aggregated white matter areas from JHU ICBM DTI atlas from FSL
# Name: (min, max)
wm_areas= {
    "Middle cerebellar peduncle": (1, 2),
    "Corpus callosum": (3, 5),
    "Fornix": (6, 6),
    "Corticospinal tract": (7, 8),
    "Medial lemniscus": (9, 10),
    "Inferior cerebellar peduncle": (11, 12),
    "Superior cerebellar peduncle": (13, 14),
    "Cerebral peduncle": (15, 16),
    "Anterior limb of internal capsule": (17, 18),
    "Posterior limb of internal capsule": (19, 20),
    "Retrolenticular part of internal capsule": (21, 22),
    "Anterior corona radiata": (23, 24),
    "Superior corona radiata": (25, 26),
    "Posterior corona radiata": (27, 28),
    "Posterior thalamic radiation": (29, 30),
    "Sagittal stratum": (31, 32),
    "External capsule": (33, 34),
    "Cingulum": (35, 38),
    "Superior longitudinal fasciculus": (41, 42),
    "Superior fronto-occipital fasciculus": (43, 44),
    "Uncinate fasciculus": (45, 46),
    "Tapetum": (47, 48),   
}
