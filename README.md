svmrs/

├── config.py                     # central CONFIG (dataset, SDCA/gossip hyperparams, thresholds)

├── main.py                       # p2pfl entry point (real gRPC gossip run)

├── peersim_run.py                # PeerSim-Python entry point (in-process simulation)

├── requirements.txt

├── .gitignore

│

├── data/

│   ├── data_loader.py            # RCV1 + covtype partitioning (p2pfl side, hf datasets)

│   └── extract_data.py           # decompress .bz2 archives into data/processed/

│

├── evaluation/

│   ├── metrics.py                # accuracy, duality gap, summary table

│   └── visualizer.py             # loss / gap / comm-cost vs time plots

│

├── baselines/

│   └── GADGET.py                 # baseline algorithm

│

├── src/

│   ├── model.py                  # SVMSDCALightning — p2pfl/Lightning SDCA model

│   └── network_layer/

│       ├── own_network/          # ── p2pfl runtime (real processes, gRPC) ──

│       │   ├── gossip_aggregator.py   # age-weighted merge (GossipAggregator)

│       │   └── network_topology.py    # node setup + topology wiring

│       └── peersim_python/       # ── PeerSim-Python engine (self-contained) ──

│           ├── __init__.py

│           ├── core.py           # Node/Network/Protocol/Linkable/CommonState/Control/Scheduler

│           ├── idle_protocol.py  # IdleProtocol (Linkable neighbour list)

│           ├── graph.py          # OverlayGraph + wireKOut

│           ├── dynamics.py       # WireKOut/Ring/Full/Star/Mesh topology controls

│           ├── cdsim.py          # CDProtocol, CDState, FullNextCycle, CDSimulator

│           ├── sdca_protocol.py  # gossip-SDCA CDProtocol (local epoch + inbox merge)

│           ├── observers.py      # DataInitializer + ConvergenceObserver (threshold stop)

│           └── logger.py         # p2pfl-free stdout logger

│

├── docs/

│   ├── README.md                 # gossip-SDCA problem + CoCoA/CoLA fix write-up

│   ├── README.pdf                # rendered PDF of the above

│   └── build_pdf.py              # markdown → PDF renderer

│

├── code-file-logic.md            # notes

└── fornow_research_talk.md       # notes
