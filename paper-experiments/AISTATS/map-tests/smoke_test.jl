# Tiny end-to-end smoke test of the harness API. Not a real experiment.
include("lv_bnode_common.jl")

prob = make_lv_problem(; σ_obs=0.5, n_train=100, n_total=200)
fns  = build_fns(prob)

# MAP (tiny) + plot
p_map, mm = run_map(prob, fns; phaseA_iters=15, phaseB_iters=15, verbose=false)
@show mm
plot_point_fit(prob, fns, p_map; outdir=ensure_outdir("smoke"), label="smoke")

# NeuralODE (tiny)
p_node, nm = run_node(prob, fns; iters=15, verbose=false)
@show nm

# NUTS (tiny) from MAP + diagnostics + posterior
samples, stats, rt = run_nuts(prob, fns, p_map; n_samples=5, n_adapts=5)
diag = nuts_diagnostics(samples, stats)
post = analyze_posterior(prob, fns, samples; outdir=ensure_outdir("smoke"), label="smoke")
@show diag post rt

append_result!(joinpath(@__DIR__, "smoke_results.csv"), (;
    arm="smoke", coverage_g=post.coverage_g, accept=diag.accept, ebfmi=diag.ebfmi,
    ess_min=diag.ess_min, rhat_max=diag.rhat_max, runtime_s=rt))

println("\nSMOKE TEST PASSED ✅")
