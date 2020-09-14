# Scripts used for custom dataset generation
## focus .py
Used for converting YOLO formatted datasets from CVAT into "focused" YOLO dataset where bbox is centered and image is padded around it by a certain predefined width in px. note that all annotations are labeled vehicle.
## focus_xml.py
Used for converting CVAT for images .xml annotations from CVAT into "focused" YOLO dataset. xml files are stored in slice_xml and are formatted as follows slice_xml/annotations{video number}.xml
## frame_slicer.py
Used to slice the videos in slice_vids into frames that are stored in slice_frames which are formatted as follows slice_frames/{video number}/{frame number with zero fill 6}.jpg. The frames generated serve as the input images for each dataset conversion script.
## xml_to_json.py
Used to convert CVAT for images .xml annotations to COCO dataset .json format. xml files are stored in slice_xml and are formatted as follows slice_xml/annotations{video number}.xml
