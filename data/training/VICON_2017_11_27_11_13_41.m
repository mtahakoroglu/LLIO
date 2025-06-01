clear all; close all; clc;
% this m file is used to examine stride indexes and correct them manually when necessary
expID = 29; % VICON room experiments of PyShoe dataset
%% import & extract data
expname = '2017-11-27-11-13-41';
filename = strcat(expname, '_LLIO.mat');
load(filename);
IMU.acc = imu_data(:,1:3);
IMU.gyro = imu_data(:,4:6);
IMU.time = timestamps_all';
% load SensorConnectData_37_LLIO.mat
% data2 = readmatrix(sprintf('SensorConnectData_%i_strideIndex_timestamp_gcpX_gcpY', expID));
% strideIndex = data2(:,1)+1; timestamps = data2(:,2); gcpX = data2(:,3); gcpY = data2(:,4);
%% visualize imu data and stride indexes
figure(expID); lw = 1.1; set(gcf, 'position', [30, 452, 1500, 267]); 
subplot(2,1,1);
plot(IMU.time, IMU.acc(:,1), 'k-');
hold on;
plot(IMU.time, IMU.acc(:,2), 'r-');
plot(IMU.time, IMU.acc(:,3), 'b-');
acc_norm = vecnorm(IMU.acc')';
plot(IMU.time, acc_norm, 'g-', 'LineWidth', 1.4);
plot(IMU.time(strideIndex), acc_norm(strideIndex), 'mx', 'LineWidth', 1.5);
% plot(IMU.time(find(IMU.time == 14.599586487000000)), ...
%     acc_norm(find(IMU.time == 14.599586487000000)), 'ko', 'LineWidth', 1.5);
% epsilon = 1e-3;
% missed_stride_timestamp = [73.5425];
% missed_stride_index = zeros(1,length(missed_stride_timestamp));
% for i=1:length(missed_stride_index)
%     missed_stride_index(i) = find(abs(IMU.time-missed_stride_timestamp(i)) <= epsilon);
% end
% missed_stride_index
grid on; set(gca, 'gridlinestyle', '--');
xlabel('Time [s]'); ylabel('Acceleration [m/s^2]');
h = legend('a_x', 'a_y', 'a_z', '||a||', 'stride');
set(h, 'location', 'southeast');
set(gca,'position', [0.0327    0.1652    0.45    0.7598]);
axis tight;
subplot(2,1,2);
plot(IMU.time, IMU.gyro(:,1), 'k-');
hold on;
plot(IMU.time, IMU.gyro(:,2), 'r-');
plot(IMU.time, IMU.gyro(:,3), 'b-');
gyro_norm = vecnorm(IMU.gyro')';
plot(IMU.time, gyro_norm, 'g-', 'LineWidth', 1.4);
plot(IMU.time(strideIndex), gyro_norm(strideIndex), 'mx', 'LineWidth', 1.5);
grid on; set(gca, 'gridlinestyle', '--'); xlabel('Time [s]');
ylabel('Angular velocity [rad/s]');
h = legend('\omega_x', '\omega_y', '\omega_z', '||\omega||', 'stride');
set(h, 'location', 'southeast');
set(gca,'position', [0.5327    0.1652    0.45    0.7598]);
axis tight;
%% plot stride indexes on acceleration and angular rate magnitude signals
figure(expID+1); clf; set(gcf, "Position", [117, 51, 1265, 318]);
% g = 9.8029; % gravity constant
plot(IMU.time, acc_norm, 'b-', 'LineWidth', lw);
hold on; plot(IMU.time, gyro_norm, 'color', [255, 127, 0]/255, 'LineWidth', lw);
plot(IMU.time(strideIndex), acc_norm(strideIndex), 'rx', 'MarkerSize', 11, 'LineWidth', lw+0.8);
% for i=1:length(missed_stride_index)
%     plot(IMU.time(missed_stride_index(i)), g*acc_norm(missed_stride_index(i)), ...
%         'go', 'MarkerSize', 10, 'LineWidth', lw+0.8);
%     plot(IMU.time(missed_stride_index(i)), gyro_norm(missed_stride_index(i)), ...
%         'go', 'MarkerSize', 10, 'LineWidth', lw+0.8);
% end
plot(IMU.time(strideIndex), gyro_norm(strideIndex), 'rx', 'MarkerSize', 11, 'LineWidth', lw+0.8);
grid on; set(gca, 'gridlinestyle', '--', 'position', [0.0469, 0.1666, 0.9448, 0.8013]);
xlabel('Time [s]', 'FontSize', 13); ylabel('Magnitude', 'FontSize', 13);
h = legend('$\Vert\mathbf{a}\Vert$', '$\Vert\mathbf{\omega}\Vert$', 'stride', 'FontSize', 15);
set(gca, 'xtick', 0:5:IMU.time(end), 'FontSize', 13);
axis tight; set(h, 'interpreter', 'latex');
strideIndexCorrection = false; % after examination assign a boolean value
if strideIndexCorrection
    print('-f',sprintf('stride_detection_exp_%i_correction', expID),'-dpng','-r600');
end