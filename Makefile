.PHONY: up down init-db run train backtest test lint

up:
	docker-compose up -d

down:
	docker-compose down

init-db:
	docker exec -i market-making-db-1 psql -U postgres -d market_making < scripts/init_db.sql

run:
	uv run python -m src.data.pipeline

train:
	uv run python scripts/train_ppo.py --total-timesteps 2000000 --n-envs 8

backtest:
	uv run python scripts/run_backtest.py --days 30

test:
	uv run pytest tests/ -v

lint:
	uv run ruff check src/ tests/ --fix
