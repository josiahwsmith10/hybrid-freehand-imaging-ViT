%% Include Necessary Directories
addpath(genpath("../radar-imaging-toolbox-private"))
addpath(genpath("./"))

%% Test with random numbers
x_in = randn(100,336) + 1j*randn(100,336);

x_out = double(pyrunfile("matlab_test_net.py","x_out",x_in=x_in,net_path="./saved/fftnet5.tar"));

%% Create the Objects
wav = TIRadarWaveformParameters();
ant = RadarAntennaArray(wav);
scanner = RadarScanner(ant);
target = RadarTarget(wav,ant,scanner);
im = RadarImageReconstruction(wav,ant,scanner,target);

%% Set Waveform Parameters
wav.f0 = 60e9;
wav.K = 124.998e12;
wav.ADCStartTime_s = 0e-6;
wav.Nk = 336;
wav.fS = 2000e3;
wav.RampEndTime_s = 2000e-6;
wav.fC = 79e9;
wav.B = 4e9;

wav.Compute();

%% Set Antenna Array Properties
ant.isEPC = false;
ant.z0_m = 0;
% Large MIMO Array
ant.tableTx = [
    0   0   1.5   5   1
    0   0   3.5   5   1];
ant.tableRx = [
    0   0   0   0   1
    0   0   0.5 0   1
    0   0   1   0   1
    0   0   1.5 0   1];
ant.Compute();

% Display the Antenna Array
ant.Display();

%% Set Scanner Parameters
scanner.method = "Rectilinear";
scanner.yStep_m = wav.lambda_m*2;
scanner.xStep_m = wav.lambda_m/4;
scanner.numY = 32;
scanner.numX = 256;

scanner.Compute();

% Display the Synthetic Array
scanner.Display();

%% Set Target Parameters
target.isAmplitudeFactor = false;

target.png.fileName = 'cutout1.png';
target.png.xStep_m = 5e-4;
target.png.yStep_m = 5e-4;
target.png.xOffset_m = 0;
target.png.yOffset_m = 0;
target.png.zOffset_m = 0.3;
target.png.reflectivity = 0.05;
target.png.downsampleFactor = 1;

target.rp.numTargets = 64;
target.rp.xMin_m = -0.2;
target.rp.xMax_m = 0.2;
target.rp.yMin_m = -0.2;
target.rp.yMax_m = 0.2;
target.rp.zMin_m = 0.1;
target.rp.zMax_m = 2;
target.rp.ampMin = 0.5;
target.rp.ampMax = 1;

% Which to use
target.isTable = false;
target.isPNG = true;
target.isSTL = false;
target.isRandomPoints = false;

target.Get();

% Display the target
target.Display();

%% Compute Beat Signal
target.isGPU = true;
target.Compute();

%% Get Dual-Band Signal
target_db = target;
s = reshape(target_db.sarData,[],wav.Nk);
s_size = size(target_db.sarData);
s(:,[1:64,(336-64+1):336]) = 0;
target_db.sarData = reshape(s,s_size);
im = RadarImageReconstruction(wav,ant,scanner,target_db);

%% Get Dual-Radar Super-Resolution Signal
target_drsr = target;
s = reshape(target_drsr.sarData,[],wav.Nk);
s_size = size(target_drsr.sarData);
s(:,[1:64,(336-64+1):336]) = 0;
s = double(pyrunfile("matlab_test_net.py","x_out",x_in=s,net_path="./saved/fftnet5.tar"));
target_drsr.sarData = reshape(s,s_size);
im = RadarImageReconstruction(wav,ant,scanner,target_drsr);

%% Set Image Reconstruction Parameters and Create RadarImageReconstruction Object 2D
im.nFFTx = 1024;
im.nFFTy = 1024;
im.nFFTz = 512;

im.xMin_m = -0.1;
im.xMax_m = 0.1;

im.yMin_m = -0.1;
im.yMax_m = 0.1;

im.zMin_m = 0.1;
im.zMax_m = 0.5;

im.numX = 200;
im.numY = 200;
im.numZ = 200;

im.isGPU = false;
im.zSlice_m = 0.3; % Use if reconstructing a 2-D image
% im.method = "Uniform 2-D SAR 3-D RMA";
im.method = "Uniform 2-D SAR 2-D FFT";

im.isMult2Mono = true;
im.zRef_m = im.zSlice_m; %0.28;

% Reconstruct the Image
im.Compute();
im.Display();


%% Save All
RadarSaveAll(wav,ant,scanner,target,im,"./matlab/results/test_fb.mat")

%% Load All
[wav,ant,scanner,target,im] = RadarLoadAll("./matlab/results/test_fb.mat");



%% Save All
RadarSaveAll(wav,ant,scanner,target,im,"./matlab/results/test_db.mat")

%% Load All
[wav,ant,scanner,target,im] = RadarLoadAll("./matlab/results/test_db.mat");



%% Save All
RadarSaveAll(wav,ant,scanner,target,im,"./matlab/results/test_drsr.mat")

%% Load All
[wav,ant,scanner,target,im] = RadarLoadAll("./matlab/results/test_drsr.mat");