
using Documenter

using InformativeSampling: Missions, Samples, SampleCosts

using InformativeSamplingUtils: DataIO, Visualization, Metrics

makedocs(
    sitename="InformativeSampling",
    # remotes=nothing,
    pages = [
        "index.md",
        "application.md",
        "core.md",
        "utilities.md",
        "code_patterns.md",
        "julia_tips.md",
    ],
    format = Documenter.HTML(prettyurls=false)
)

deploydocs(
    repo = "github.com/ngharrison/InformativeSampling.git",
    versions = nothing,
)
