import glob
import os
import shutil
import sys
from typing import Any, Dict, List

from argoverse.utils.json_utils import read_json_file, save_json_dict

root_dir = sys.argv[1]

print("root dir = ", root_dir)
print("updating track_labels_amodal folders...")
list_log_folders = glob.glob(os.path.join(root_dir, "*"))

if len(list_log_folders) == 0:
    print("Not file founded.")
else:
    for ind_log, path_log in enumerate(list_log_folders):
        print("Processing %d/%d" % (ind_log + 1, len(list_log_folders)))
        list_path_label_persweep = glob.glob(os.path.join(path_log, "per_sweep_annotations_amodal", "*"))
        list_path_label_persweep.sort()
        dist_track_labels: Dict[str, List[Any]] = {}
        for path_label_persweep in list_path_label_persweep:
            data = read_json_file(path_label_persweep)
            for data_obj in data:
                id_obj = data_obj["track_label_uuid"]
                if id_obj not in dist_track_labels.keys():
                    dist_track_labels[id_obj] = []
                dist_track_labels[id_obj].append(data_obj)

        path_amodal_labels = os.path.join(path_log, "track_labels_amodal")
        data_amodal: Dict[str, Dict[str, Any]] = {}

        if os.path.exists(path_amodal_labels):
            shutil.rmtree(path_amodal_labels)

        os.mkdir(path_amodal_labels)
        print("Adding files to ", path_amodal_labels)
        for key in dist_track_labels.keys():
            data_amodal[key] = {
                "label_class": dist_track_labels[key][0]["label_class"],
                "uuid": dist_track_labels[key][0]["track_label_uuid"],
                "log_id": path_log.split("/")[-1],
                "track_label_frames": dist_track_labels[key],
            }
            save_json_dict(os.path.join(path_amodal_labels, "%s.json" % key), data_amodal[key])
