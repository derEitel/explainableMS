import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

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

def generate_transparanet_cm(base='coolwarm', name="TransCoWa"):
    # copy from existing colormap
    ncolors = 256
    color_array = plt.get_cmap(base)(range(ncolors))

    # create parabolic decrease 
    decr = [-1*(x**2)+1 for x in range(int(ncolors/2))]
    # normalize
    decr = (decr - np.min(decr))/(np.max(decr - np.min(decr)))

    # use inverted parabola as increase
    incr = np.copy(decr)[::-1]
    alphas = np.concatenate((decr, incr))
    # update alpha values
    color_array[:,-1] = alphas

    # create new colormap and register it
    transparent_coolwarm = LinearSegmentedColormap.from_list(name, color_array)
    plt.register_cmap(cmap=transparent_coolwarm)

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
        # sort by values
        area_hm_sorted = sorted(area_hm.items(), key=lambda kv: kv[1])
        keys_sorted = [row[0] for row in area_hm_sorted]
        values_sorted = [row[1] for row in area_hm_sorted]
        
        keys.append(keys_sorted)
        values.append(values_sorted)
    return keys, values

def translate_keys(keys):
    names_list = []
    for key_list in keys:
        name_list = []
        for key in key_list:
            name_list.append(short_name_map[key])
        names_list.append(name_list)
    return names_list

def wrap_as_df(keys, values):
    df_ms = pd.DataFrame({"values_ms": values[0]}, keys[0])
    df_hc = pd.DataFrame({"values_hc": values[1]}, keys[1])

    df = pd.merge(df_ms, df_hc, left_index=True, right_index=True, how='outer')
    return df

def reduce_df(df, take=30):
    # get order based on relevance sum
    abs_order = (np.abs(df["values_hc"]) + np.abs(df["values_ms"])).sort_values().index
    most = abs_order[-take:]
    short_df = df.loc[most]
    
    order = (short_df["values_hc"] + short_df["values_ms"]).sort_values().index
    short_df = df.loc[order]
    return short_df

def reduce_two_dfs(df_zero, df_one, take=30):
    abs_order = (df_zero.abs().sum() + df_one.abs().sum()).sort_values().index
    most = abs_order[-take:]

    # columns are keys so use [:, key]
    short_df_zero = df_zero.loc[:,most]
    short_df_one = df_one.loc[:,most]
    
    order = (short_df_zero.sum() + short_df_one.sum()).sort_values().index
    short_df_zero = short_df_zero.reindex(order, axis=1)
    short_df_one = short_df_one.reindex(order, axis=1)
    return short_df_zero, short_df_one

def plot_key_value_pairs(keys, values, title, loc="center left"):
    plt.figure(figsize=(10, 6))
    plt.plot(keys[0], values[0], 'o', color=ms_color, label="CDMS")
    plt.plot(keys[1], values[1], 'o', color=hc_color, label="HC")
    plt.xticks(rotation='vertical')
    plt.legend(loc=loc)
    plt.title(title)
    plt.show()

def plot_dataframe(df, title, loc="center left"):
    plt.figure(figsize=(10, 6))
    plt.plot(df["values_ms"], 'o', color=ms_color, label="CDMS")
    plt.plot(df["values_hc"], 'o', color=hc_color, label="HC")
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
    "Hippocampus": (47, 48),
    "Parahippocampal gyrus": (170, 171),
    "Pallidum": (55, 56),
    "Putamen": (57, 58),
    "Thalamus": (59, 60),
    "CWM": (44, 45),
    "ACG": (100, 101),
    "Ant. Insula": (102, 103),
    "Post. Insula": (172, 173),
    "AOG": (104, 105),
    "AG": (106, 107),
    "Cuneus": (114, 115),
    "Central operculum": (112, 113),
    "Frontal operculum": (118, 119),
    "Frontal pole": (120, 121),
    "Fusiform gyrus": (122, 123),
    "Temporal pole": (202, 203),
    "TrIFG": (204, 205),
    "TTG": (206, 207),
    "Entorh. cortex": (116, 117),
    "Parietal operculum": (174, 175),
    "SPL": (198, 199),
    "CSF": (46, 46),
    "3rd Ventricle": (4, 4),
    "4th Ventricle": (11, 11),
    "Lateral Ventricles": (49, 52),
    "Diencephalon": (61, 62),
    "Vessels": (63, 64),
    "Optic Chiasm": (69, 69),
    "Vermal Lobules": (71, 73),
    "Basal Forebrain": (75, 76),
    "Calc": (108, 109),
    "GRe": (124, 125),
    "IOG": (128, 129),
    "ITG": (132, 133),
    "LiG": (134, 135),
    "LOrG": (136, 137),
    "MCgG": (138, 139),
    "MFC": (140, 141),
    "MFG": (142, 143),
    "MOG": (144, 145),
    "MOrG": (146, 147),
    "MPoG": (148, 149),
    "MPrG": (150, 151),
    "MSFG": (152, 153),
    "MTG": (154, 155),
    "OCP": (156, 157),
    "OFuG": (160, 161),
    "OpIFG": (162, 163),
    "OrIFG": (164, 165),
    "PCgG": (166, 167),
    "PCu": (168, 169),
    "PoG": (176, 177),
    "POrG": (178, 179),
    "PP": (180, 181),
    "PrG": (182, 183),
    "PT": (184, 185),
    "SCA": (186, 187),
    "SFG": (190, 191),
    "SMC": (192, 193),
    "SMG": (194, 195),
    "SOG": (196, 197),
    "STG": (200, 201),
}

short_name_map = {
     'Accumbens': 'Accumbens',
     'Amygdala': 'Amygdala',
     'Brain Stem': 'Brain Stem',
     'Caudate': 'Caudate',
     'Cerebellum': 'Cerebellum',
     'Hippocampus': 'Hippocampus',
     'Parahippocampal gyrus': 'Parahippocampal gyr.',
     'Pallidum': 'Pallidum',
     'Putamen': 'Putamen',
     'Thalamus': 'Thalamus',
     'Diencephalon': 'Diencephalon',
     'CWM': 'Cerebral white matter',
     'ACG': 'Ant. cingulate gyr.',
     'Ant. Insula': 'Ant. insula',
     'Post. Insula': 'Post. insula',
     'AOG': 'Ant. orbital gyr.',
     'AG': 'Angular gyr.',
     'Cuneus': 'Cuneus',
     'Central operculum': 'Central operculum',
     'Frontal operculum': 'Frontal operculum',
     'Frontal pole': 'Frontal pole',
     'Fusiform gyrus': 'Fusiform gyr.',
     'Temporal pole': 'Temporal pole',
     'TrIFG': 'Triangular part of IFG',
     'TTG': 'Trans. temporal gyr.',
     'Entorh. cortex': 'Entorhinal area',
     'Parietal operculum': 'Parietal operculum',
     'SPL': 'Sup. parietal lobule',
     'CSF': 'CSF',
     '3rd Ventricle': '3rd Ventricle',
     '4th Ventricle': '4th Ventricle',
     'Lateral Ventricles': 'Inf. Lat. Ventricles',
     'Vessels': 'Vessels',
     'Optic Chiasm': 'Optic Chiasm',
     'Vermal Lobules': 'Cereb. Verm. Lob.',
     'Basal Forebrain': 'Basal Forebrain',
     'Calc': 'Calcarine cortex',
     'GRe': 'Gyrus rectus',
     'IOG': 'Inf. occipital gyr.',
     'ITG': 'Inf. temporal gyr.',
     'LiG': 'Lingual gyr.',
     'LOrG': 'Lat. orbital gyr.',
     'MCgG': 'Mid. cingulate gyr.',
     'MFC': 'Med. frontal cortex',
     'MFG': 'Mid. frontal gyr.',
     'MOG': 'Mid. occipital gyr.',
     'MOrG': 'Med. orbital gyr.',
     'MPoG': 'Post. gyr. med. seg.',
     'MPrG': 'Pre. gyr. med. seg.',
     'MSFG': 'Sup. frontal gyr. med. seg.',
     'MTG': 'Mid. temporal gyr.',
     'OCP': 'Occipital pole',
     'OFuG': 'Occipital fusiform gyr.',
     'OpIFG': 'Opercular part of IFG',
     'OrIFG': 'Orbital part of IFG',
     'PCgG': 'Post. cingulate gyr.',
     'PCu': 'Precuneus',
     'PoG': 'Postcentral gyr.',
     'POrG': 'Post. orbital gyr.',
     'PP': 'Planum polare',
     'PrG': 'Precentral gyr.',
     'PT': 'Planum temporale',
     'SCA': 'Subcallosal area',
     'SFG': 'Sup. frontal gyr.',
     'SMC': 'Supp. motor cortex',
     'SMG': 'Supramarginal gyr.',
     'SOG': 'Sup. occipital gyr.',
     'STG': 'Sup. temporal gyr.'
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
