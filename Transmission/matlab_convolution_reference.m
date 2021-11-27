clear all; clc;

f = rcosdesign(0.3, 4, 4)';

rand('twister', 712);
signal = rand(100, 1);

y_conv = conv(signal, f)

y_filter = filter(f, 1, signal)
y_filter(1:10)
