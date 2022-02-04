saved files

hffh_ViT_solid_w_points_2048_training.mat
- X-Y scanning X-Y image: freehand scenario
- Planar/rectilinear array with 25 Y steps (200 virtual Y antenna locations) and 200 X steps (200x200 array)
	- Ideal perfectly placed antennas
- fmcw_v2 chirp parameters (64 ADC samples, etc.)
- o_x = o_y = 1.2 mm
- no amplitude terms
- randomly placed points AND solid shapes placed and shaped randomly
- (200, 200) = (X, Y)
	- radarImages (features) - (200, 200, 2048) = (X, Y, sample index)
	- idealImages (labels)	 - (200, 200, 2048) = (X, Y, sample index) 
- X and Y are between [-0.1, 0.1) m
