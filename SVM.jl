using DataFrames
using Distributions
using Plots
pyplot()


function transform_svm_primal(X, y, C = .5)
    d = size(X,2)
    n = size(X,1)
    Q = zeros(d+n, d+n)
    Q[1:d, 1:d] = eye(d)
    p = zeros(d+n)
    p[d+1:d+n] = [C for _ in 1:n]
    Y = diagm(y[:Label])
    A = zeros(n, d+n)
    A[1:n,1:d] = - Y*Array(X)
    A[1:n, d+1:d+n] = -eye(n)
    b = -ones(n)
    return Q, p, A, b
end


function transform_svm_dual( X, y, C = .5)
    d = size(X,2)
    n = size(X,1)
    Q = diagm(y[:Label]) * Array(X) * transpose(Array(X)) * diagm(y[:Label])
    p = ones(n)
    A = zeros(2*n, n)
    b = zeros(2*n)
    A[1:n, 1:n] = eye(n)
    A[n+1:2n, 1:n] = -eye(n)
    b[1:n] = C*ones(n)
    return Q, p, A, b
end


function generate_data(n1, n2, d=2)
    # Returns 2 clouds of labelled Gaussian data of dim 2 with different moments
    model1 = MvNormal(-50*ones(d), .5 * eye(d))
    model2 = MvNormal(50*ones(d), .5 * eye(d))
    samples1 = rand(model1, n1)
    samples2 = rand(model2, n2)
    X = DataFrame([transpose(samples1); transpose(samples2)])
    X = hcat(X, zeros(n1+n2))
    y = DataFrame(Label=ones(size(X,1)))
    y[n1+1:n1+n2,:Label] = -1*ones(n2)
    return X, y
end


function backtracking(x, t, dx, f, nabla)
    alpha = .25 
    beta = .5
    res = 0.1
    while minimum(x + res*dx) < 10^(-3.)
        res *= beta
    end
    k= 0
    println(dx)
    val = f(x,t)
    next_val = f(x+res*dx,t)
    while  next_val> val + (alpha * res*transpose(nabla)*dx)[1]
        println(res*(transpose(nabla)*dx)[1])
        k+=1
        res *= beta
        next_val = f(x+res*dx,t)
        if k % 100 == 0
            println(k)
            println(val)
            println(next_val)
            println((alpha * res*transpose(nabla)*dx)[1])
        end
    end
    return res
end

function Newton(t, x, f, grad, hess)
    nabla = grad(x,t)
    invHess = inv(hess(x,t) + 10^(-3.) * eye(size(nabla,1)))
    dx = - invHess * nabla
    lam2 = - transpose(dx) * nabla
    backtrack_t = backtracking(x, t, dx, f, nabla)
    x_new = x + backtrack_t*dx
    gap = lam2/2
    return x_new, gap[1]
end

function centering_step(Q, p, A, b, x, t=1, tol=10^(-3.))
    function f(u,v)
        return (v * (transpose(u) * Q * u + transpose(p)*u) - sum(log(b - A * u)))[1]
    end
    function grad(u,v)
        return v * (Q * u + p) + sum([transpose(A[i,:])/(b[i] - A[i,:] * u)[1] for i in 1:size(A,1)])
    end
    function hess(u,v)
        return v * Q - sum([transpose(A[i,:])*A[i,:]/(b[i] - A[i,:] * u)[1]^2. for i in size(A,1)])
    end
    x_seq = []
    gaps = []
    obj_vals = []
    gap = tol + 1
    x_new = x
    j = 0
    while (gap > tol)
        j += 1
        @printf "Newton %d \n" j
        x_new, gap = Newton(t, x_new, f, grad, hess)
        println(gap)
        push!(x_seq, x_new)
        push!(gaps, gap)
        push!(obj_vals, f(x_new,t))
        #plot(gaps)
        #plot(x_seq)
        #plot(obj_vals)
    end
    println(gap)
    return x_seq
end

function barr_method(Q, p, A, b, x_0, mu, tol=10^(-3.))
    x_sol = x_0
    m = size(x_0, 1)
    t = .5
    gaps = []
    k = 0
    while m/t > tol   
        k+= 1
        @printf "Barrier %d \n" k
        push!(gaps, m/t)
        println(m/t)
        x_sol = centering_step(Q, p, A, b, x_sol, t, tol)[end]
        t = mu * t
        plot(gaps)
    end
    return x_sol, x_seq
end


function SVM_primal(X, y, C = .5)
    n, d = size(X)
    Q, p, A, b = transform_svm_primal(X,y,C)
    mu = 2
    x_0 = C/2 * ones(size(A,2))
    barr_method(Q,p,A,b, x_0, mu)
end


function SVM_dual(X, y, C=.5)
    Q, p, A, b = transform_svm_dual(X, y,C)
    mu = 2
    x_0 = C/2 * ones(size(A,2))
    barr_method(Q,p,A,b, x_0, mu)
end

X,y = generate_data(1,1)
C = .5
SVM_dual(X,y,C)

# X,y = generate_data(20,20)
# C = .5
# Q, p, A, b = transform_svm_dual(X,y,C)
# x= C/2 * ones(size(A,2))
# function f(u,v)
#     return v * (transpose(u) * Q * u + transpose(p)*u)[1]
# end
# f(x,1)




