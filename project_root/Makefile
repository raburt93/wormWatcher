
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

