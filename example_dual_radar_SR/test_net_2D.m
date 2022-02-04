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
scanner.method = "Linear";
scanner.yStep_m = wav.lambda_m*2;
scanner.numY = 32;

scanner.Compute();

% Display the Synthetic Array
scanner.Display();

%% Set Target Parameters
target.isAmplitudeFactor = false;

target.tableTarget = [
    0   0       0.25    1
    0   0.1     0.25    1];

X = [1,1,1,2,3,3,3,5,6,6,6,6,7,9,9,9,9,10,10,11,11]'*15e-3;
X = X - mean(X);
Y = [6,5,4,3,4,5,6,6,6,5,4,3,6,6,5,4,3,6,3,5,4]'*45e-3;
Y = Y - mean(Y);

% UTD
target.tableTarget = [zeros(size(X)),X,Y + 300e-3,ones(size(X))];

% Which to use
target.isTable = true;
target.isPNG = false;
target.isSTL = false;
target.isRandomPoints = false;

target.Get();

% Display the target
target.Display();

%% Get Full-Band Signal
target.isGPU = true;
target.Compute();
im = RadarImageReconstruction(wav,ant,scanner,target);

%% Get Dual-Band Signal
s = reshape(target.sarData,[],wav.Nk);
s_size = size(target.sarData);
s(:,65:(336-64)) = 0;
target.sarData = reshape(s,s_size);
im = RadarImageReconstruction(wav,ant,scanner,target);

%% Get Dual-Radar Super-Resolution Signal
s = reshape(target.sarData,[],wav.Nk);
s_size = size(target.sarData);
s(:,65:(336-64)) = 0;
s = test_net_sarData(s,"./saved/fftnet5.tar");
target.sarData = reshape(s,s_size);
im = RadarImageReconstruction(wav,ant,scanner,target);

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
im.method = "Uniform 1-D SAR 2-D RMA";

im.isMult2Mono = true;
im.zRef_m = im.zSlice_m; %0.28;

% Reconstruct the Image
im.Compute();

im.dBMin = -15;
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