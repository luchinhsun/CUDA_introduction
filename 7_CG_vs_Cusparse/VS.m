clear
x = load('cusparse/X_ans.txt');
x_CG = load('CG/X_CG.txt');

%A = load('CG/A.txt');
%B = load('CG/B.txt');

N = 10000;
A = -2*eye(N);
B = zeros(N,1);
B(1) = -1;
for i = (1:N-1)
    A(i,i+1) = 1;
    A(i+1,i) = 1;
end

max(abs(A*x - B))
max(abs(A*x_CG - B))
