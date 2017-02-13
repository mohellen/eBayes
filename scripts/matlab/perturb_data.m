
infile = '../simulations/ns_400x80_obs4/data.dat';
outfile = '../simulations/ns_400x80_obs4/data_10percent.dat';
sigma = 0.1;

display('Perturbing data set: obs4');

d = read_file(infile);

% make data with noise
sig = mean(d) * sigma;
dn = d + normrnd(0,sig,size(d));

per = abs(d-dn)./d;
for i = 1:size(d,1),
   sig2 = sig;
   while per(i) > sigma,
       sig2 = sig2/2; 
       dn(i) = d(i) + normrnd(0,sig2);
       per(i) = abs(d(i) - dn(i))/d(i);
   end
end

per = abs(d-dn)./d;
[d,dn,per]

write_file(dn, outfile)