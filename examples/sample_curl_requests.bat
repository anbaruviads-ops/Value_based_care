@echo off
echo Testing AI Assistant Health Endpoint...
curl -X GET http://localhost:8005/health

echo.
echo Testing Direct Financial Query...
curl -X POST http://localhost:8005/api/assistant/ask ^
  -H "Content-Type: application/json" ^
  -d "{\"question\": \"What was ACO A1001's savings rate and expenditure in 2021?\", \"context\": {\"aco_id\": \"A1001\", \"year\": 2021, \"page\": \"ACO Explorer\"}}"

echo.
echo Testing What-If Scenario Query...
curl -X POST http://localhost:8005/api/assistant/ask ^
  -H "Content-Type: application/json" ^
  -d "{\"question\": \"What happens under this scenario?\", \"context\": {\"aco_id\": \"A1001\", \"scenario\": {\"metric\": \"ER visits\", \"change\": \"-5%%\"}, \"result\": {\"projected_expenditure_change\": \"-2.1%%\"}}}"
