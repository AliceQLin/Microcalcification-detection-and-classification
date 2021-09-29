import csv

# return annotation dicts {img_name: {"bbx": [N * 4], "labels": [N]}}
def csv_to_label_and_bbx(csv_file_path):
    class_ids = {"0": 1, "1": 2}
    stats = {"0": 0, "1": 0}
    anno_dicts = {}
    with open(csv_file_path, newline='', encoding='UTF-8') as csvfile:
        rows = csv.reader(csvfile)
        rows = list(rows)
        for row in rows[1:]:
            anno_dict = {}
            img_name = row[0]
#            print(img_name)
            raw_boxes = row[1:]
            bounding_boxes = []
            labels = []
            for box in raw_boxes:
                if box == '':
                    continue
                dic_box = eval(box)
                label = dic_box["label"]
#                print(label)
                y = dic_box["y"]
                x = dic_box["x"]
                height = dic_box["height"]
                width = dic_box["width"]
                xmin = x
                xmax = x + width
                ymin = y
                ymax = y + height
                stats[label] += 1
                label = class_ids[label]
                bounding_boxes.append([xmin, ymin, xmax, ymax])
                labels.append(label)
            anno_dict["bbx"] = bounding_boxes
            anno_dict["labels"] = labels
            anno_dicts[img_name] = anno_dict

    print(stats)
    return anno_dicts

