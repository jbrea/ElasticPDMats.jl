using ElasticPDMats, PDMats, LinearAlgebra
using Test

@testset "PDMat Tests" begin
    a = rand(10, 10); m = a*a';
    epdmat = ElasticPDMat(m, capacity = 2000);
    test_pdmat(epdmat, m, verbose = 0)
end

@testset "append!" begin
    a = rand(10, 10); m = a*a';
    epdmat = ElasticPDMat(m[1:9, 1:9])
    append!(epdmat, m[:, 10])
    @test cholesky(m).U ≈ view(epdmat.chol).U

    epdmat = ElasticPDMat(m[1:6, 1:6])
    append!(epdmat, m[:, 7:10])
    @test cholesky(m).U ≈ view(epdmat.chol).U

    epdmat = ElasticPDMat(m[1:6, 1:6], capacity = 6, stepsize = 6)
    append!(epdmat, m[:, 7:10])
    @test epdmat.mat.capacity == epdmat.chol.capacity == 
        size(epdmat.mat.data, 1) == size(epdmat.chol.c.factors, 1) == 12
    @test cholesky(m).U ≈ view(epdmat.chol).U
end

@testset "deleteat!" begin
    a = rand(10, 10); m = a*a';
    epdmat = ElasticPDMat(m)
    deleteat!(epdmat, 3)
    m2 = m[[1:2; 4:10], [1:2; 4:10]]
    @test cholesky(m2).U ≈ view(epdmat.chol).U

    a = rand(10, 10); m = a*a';
    epdmat = ElasticPDMat(m)
    deleteat!(epdmat, [3, 8, 7])
    m2 = m[[1:2; 4:6; 9:10], [1:2; 4:6; 9:10]]
    @test cholesky(m2).U ≈ view(epdmat.chol).U
end
