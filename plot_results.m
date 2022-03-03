%% Include Necessary Directories
addpath(genpath("../radar-imaging-toolbox-private"))
addpath(genpath("./"))
set(0,'DefaultFigureWindowStyle','docked')

%% Test Example (707)
load test707.mat

%% Create Scenario
x_m = linspace(-0.1,0.1,256);
y_m = linspace(-0.1,0.1,256);

%% Plot LR Image
figure(1)
plotXYdB(lr,x_m,y_m,-25,"x (m)","y (m)","",25)
saveas(gcf,"./results/test707_lr.jpg")
saveas(gcf,"./results/test707_lr.fig")

%% Plot HR Image
figure(2)
plotXYdB(hr,x_m,y_m,-25,"x (m)","y (m)","",25)
saveas(gcf,"./results/test707_hr.jpg")
saveas(gcf,"./results/test707_hr.fig")

%% Plot SR Image
figure(3)
plotXYdB(sr,x_m,y_m,-25,"x (m)","y (m)","",25)
saveas(gcf,"./results/test707_sr.jpg")
saveas(gcf,"./results/test707_sr.fig")


%% Test Example (270)
load test270.mat

%% Create Scenario
x_m = linspace(-0.1,0.1,256);
y_m = linspace(-0.1,0.1,256);

%% Plot LR Image
figure(4)
plotXYdB(lr,x_m,y_m,-25,"x (m)","y (m)","",25)
saveas(gcf,"./results/test270_lr.jpg")
saveas(gcf,"./results/test270_lr.fig")

%% Plot HR Image
figure(5)
plotXYdB(hr,x_m,y_m,-25,"x (m)","y (m)","",25)
saveas(gcf,"./results/test270_hr.jpg")
saveas(gcf,"./results/test270_hr.fig")

%% Plot SR Image
figure(6)
plotXYdB(sr,x_m,y_m,-25,"x (m)","y (m)","",25)
saveas(gcf,"./results/test270_sr.jpg")
saveas(gcf,"./results/test270_sr.fig")


%% Test Example (102)
load test102.mat

%% Create Scenario
x_m = linspace(-0.1,0.1,256);
y_m = linspace(-0.1,0.1,256);

%% Plot LR Image
figure(7)
plotXYdB(lr,x_m,y_m,-25,"x (m)","y (m)","",25)
saveas(gcf,"./results/test102_lr.jpg")
saveas(gcf,"./results/test102_lr.fig")

%% Plot HR Image
figure(8)
plotXYdB(hr,x_m,y_m,-25,"x (m)","y (m)","",25)
saveas(gcf,"./results/test102_hr.jpg")
saveas(gcf,"./results/test102_hr.fig")

%% Plot SR Image
figure(9)
plotXYdB(sr,x_m,y_m,-25,"x (m)","y (m)","",25)
saveas(gcf,"./results/test102_sr.jpg")
saveas(gcf,"./results/test102_sr.fig")



%% Real Image
load test_exp1.mat

%% Plot LR Image
figure(10)
plotXYdB(lr,x_m,y_m,-25,"x (m)","y (m)","",25)
saveas(gcf,"./results/exp1_lr.jpg")
saveas(gcf,"./results/exp1_lr.fig")

%% Plot SR Image
figure(11)
plotXYdB(sr,x_m,y_m,-25,"x (m)","y (m)","",25)
saveas(gcf,"./results/exp1_sr.jpg")
saveas(gcf,"./results/exp1_sr.fig")