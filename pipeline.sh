# $1 : Output file
# $2 : Output file
transform="identity"
python get_roi_video.py "$1" "${1}.roi.avi" --transform ${transform}

# Get the pose information
cd openpose/
build/examples/openpose/openpose.bin --render_pose 1 \
				     --face_render 0 \
				     --hand_render 0 \
				     -number_people_max 1\
				     --video ${1}.roi.avi \
				     --display 0 \
				     -write_json ${1}.roi.json \
				     -write_images ${1}.roi.out_image


cd -
# Get what each pose means
#classifier.py ${1} json.${2} ${1}.processed.avi
