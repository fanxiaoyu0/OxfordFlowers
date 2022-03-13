import neptune.new as neptune

run = neptune.init(
    project="fanxiaoyu1234/OxfordFlowers",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0\
    cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJhNTEyOTJhYi0zN2E2LTQzMWQtODI3ZC1iNWFjY2M2NDdjMmUifQ==",
)  # your credentials

params = {"learning_rate": 0.001, "optimizer": "Adam"}
run["parameters"] = params

for epoch in range(10):
    run["train/loss"].log(0.9 ** epoch)

run["eval/f1_score"] = 0.66

run.stop()