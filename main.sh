cd openpose
build/examples/openpose/openpose.bin --render_pose 0 \
				     --face_render 0 \
				     --hand_render 0 \
				     -keypoint_scale 3\
				     --video $1 \
				     --display 0 \
				     -write_json $2



cd -
