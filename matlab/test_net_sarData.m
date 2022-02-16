function s = test_net_sarData(s,netpath,N)

if nargin < 3
    N = 64;
end

s_norm = sqrt(1/128 * sum(abs(s).^2,2));

s = fft(s./s_norm,[],2);

for indn = 1:ceil(size(s,1)/N)
    indStart = N*(indn-1)+1;
    indEnd = min([size(s,1),N*indn]);
    s(indStart:indEnd,:) = double(pyrunfile("matlab_test_net.py","x_out",x_in=s(indStart:indEnd,:),net_path=netpath));
end

s = s .* s_norm;
