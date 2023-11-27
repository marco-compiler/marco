module {
 func.func @test() {
    %0 = sbg.node {id = 0, elems = #sbg<set {[1:1:10]x[1:1:10]}>} : !sbg.node
    func.return
 }
}