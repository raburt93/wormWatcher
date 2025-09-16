
.PHONY: ingest diag
ingest: ## Ingest a video: make ingest VIDEO=videos/YourFile.m4v
	@if [ -z "$(VIDEO)" ]; then echo "Usage: make ingest VIDEO=videos/YourFile.m4v"; exit 2; fi
	python -m src.ingest config.yaml $(VIDEO)

.PHONY: verify
verify: ## Verify hash + constant FPS (needs ffprobe for best results)
	@if [ -z "$(VIDEO)" ]; then echo "Usage: make verify VIDEO=videos/YourFile.m4v"; exit 2; fi
	python -m src.verify_fps config.yaml $(VIDEO)

diag: ## Scan logs/manifests for issues
	python -m src.diag config.yaml
	@if [ -z "$(VIDEO)" ]; then echo "Usage: make ingest VIDEO=videos/YourFile.m4v"; exit 2; fi
	python -m src.ingest config.yaml $(VIDEO)

.PHONY: preprocess detect_track qc pipeline

preprocess: ## Denoise + background model preview + motion energy
	@if [ -z "$(VIDEO)" ]; then echo "Usage: make preprocess VIDEO=videos/YourFile.m4v"; exit 2; fi
	python -m src.preprocess config.yaml $(VIDEO)

detect_track: ## Segment single worm in ROI and write tracks table
	@if [ -z "$(VIDEO)" ]; then echo "Usage: make detect_track VIDEO=videos/YourFile.m4v"; exit 2; fi
	python -m src.detect_track config.yaml $(VIDEO)

qc: ## Draw overlay video/snapshots from tracks
	@if [ -z "$(VIDEO)" ]; then echo "Usage: make qc VIDEO=videos/YourFile.m4v"; exit 2; fi
	python -m src.qc config.yaml $(VIDEO)

pipeline: ## ingest -> preprocess -> detect_track -> qc
	@if [ -z "$(VIDEO)" ]; then echo "Usage: make pipeline VIDEO=videos/YourFile.m4v"; exit 2; fi
	make ingest VIDEO=$(VIDEO) && make preprocess VIDEO=$(VIDEO) && make detect_track VIDEO=$(VIDEO) && make qc VIDEO=$(VIDEO)

.PHONY: split_audio
split_audio: ## Detect 50 trials from audio (>=600s), validate light, cut 8s frame-accurate clips
	@if [ -z "$(VIDEO)" ]; then echo "Usage: make split_audio VIDEO=videos/YourFile.m4v"; exit 2; fi
	python -m src.split_audio config.yaml $(VIDEO) --min-start-s 600 --expected 50 --trial-dur 6 --lead 1 --tail 1 --reencode

.PHONY: split_unpaired
split_unpaired: ## Classify trials per schedule (audio/light/paired/unpaired) and cut 8s clips
	@if [ -z "$(VIDEO)" ]; then echo "Usage: make split_unpaired VIDEO=videos/YourFile.m4v [SCHED=events/sched.csv] or grid vars"; exit 2; fi
	@if [ -n "$(SCHED)" ]; then \
		python -m src.split_unpaired config.yaml $(VIDEO) --schedule-csv $(SCHED) --min-start-s 600 --trial-dur 6 --lead 1 --tail 1 --pre-win 2 --post-win 8 --pair-tol 0.6 --expected-light-offset 4.0; \
	else \
		python -m src.split_unpaired config.yaml $(VIDEO) --grid-t0 611 --grid-period 12 --grid-count 50 --min-start-s 600 --trial-dur 6 --lead 1 --tail 1 --pre-win 2 --post-win 8 --pair-tol 0.6 --expected-light-offset 4.0; \
	fi
