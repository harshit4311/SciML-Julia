#=
plot_helpers.jl — legible plotting for the dense, many-cycle chemostat series.

The raw C1 record is ~360 points over ~8–9 predator–prey cycles. Cramming that
into one axis (and overlaying a posterior cloud) hides the data. `faceted_grid`
instead lays out a grid of (time-window rows × channel columns): each panel zooms
one channel over one slice of time, with the data drawn *on top* of any posterior
band so it is always visible.

Used by both plot_data.jl (raw) and algae_chemostat.jl (with posterior overlay).
=#

import Plots

# Contiguous equal-time windows over the day axis; returns per-window boolean
# masks plus the window edges.
function _window_masks(day::AbstractVector, nwin::Int)
    d0, d1 = minimum(day), maximum(day)
    edges = collect(range(d0, d1, length=nwin + 1))
    masks = Vector{BitVector}(undef, nwin)
    for w in 1:nwin
        lo, hi = edges[w], edges[w + 1]
        masks[w] = w == nwin ? ((day .>= lo) .& (day .<= hi)) :
                               ((day .>= lo) .& (day .< hi))
    end
    return masks, edges
end

"""
    faceted_grid(outpath, day, channels; labels, colors, kwargs...)

Save a (nwin × nchannels) panel grid. `channels` is a vector of equal-length
data vectors aligned to `day`. Each row is a time window (auto y-zoom), each
column a channel. Data is drawn on top as solid markers + a faint connector.

Optional `bands` = vector of `(lo, hi, mean)` triples (one per channel, aligned
to `day`) draws a light posterior-predictive ribbon *underneath* the data.
`split_day` draws a train/forecast divider where it falls inside a window.
"""
function faceted_grid(outpath::String, day::AbstractVector, channels::Vector;
                      labels::Vector{String}, colors::Vector,
                      nwin::Int=4, split_day=nothing, bands=nothing,
                      title::String="", ms::Real=2.6)
    nchan = length(channels)
    masks, edges = _window_masks(day, nwin)
    panels = Vector{Any}(undef, nwin * nchan)

    for w in 1:nwin, c in 1:nchan
        m = masks[w]
        idx = (w - 1) * nchan + c
        p = Plots.plot(legend=false, grid=false,
                       title = w == 1 ? labels[c] : "", titlefontsize=9,
                       ylabel = c == 1 ? "day $(round(Int, edges[w]))–$(round(Int, edges[w+1]))" : "",
                       xlabel = w == nwin ? "day" : "", xtickfontsize=6, ytickfontsize=6,
                       left_margin=2Plots.mm)

        # Posterior band underneath (finite points only, so failed solves break
        # the ribbon instead of erroring).
        if bands !== nothing
            lo, hi, mn = bands[c]
            bm = m .& isfinite.(lo) .& isfinite.(hi) .& isfinite.(mn)
            if any(bm)
                Plots.plot!(p, day[bm], mn[bm],
                            ribbon=(mn[bm] .- lo[bm], hi[bm] .- mn[bm]),
                            fillalpha=0.22, color=colors[c], lw=1.4, alpha=0.9)
            end
        end

        # Data on top: faint connector + solid markers.
        x, y = day[m], channels[c][m]
        if any(m)
            Plots.plot!(p, x, y, lw=0.6, color=:gray65, alpha=0.5)
            Plots.scatter!(p, x, y, ms=ms, mc=colors[c], msw=0)
        end

        if split_day !== nothing && edges[w] <= split_day <= edges[w + 1]
            Plots.vline!(p, [split_day], color=:black, ls=:dash, lw=1)
        end
        panels[idx] = p
    end

    fig = Plots.plot(panels...; layout=(nwin, nchan),
                     size=(640 * nchan, 200 * nwin),
                     plot_title=title, plot_titlefontsize=11)
    Plots.savefig(fig, outpath)
    return outpath
end
