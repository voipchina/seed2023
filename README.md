## STEP1: Aliyun Build Docker (about 132sec)
https://cr.console.aliyun.com/repository/cn-hangzhou/mario_odyssey/seed2023/build


## Push Locally
docker push registry.cn-hangzhou.aliyuncs.com/mario_odyssey/seed2023:11051438 

## STEP2: Submit Predicts (系统评测时间为10:00、16:00)
https://www.marsbigdata.com/competition/details?id=40144198635


## Verify
curl -X 'POST'   'http://localhost:8000/mask' \
 -H 'accept: application/json' \
 -H 'Content-Type: multipart/form-data' \
 -F 'file=@/Users/steven/my/code/2024SeedThyroid/datasets/origin/valid1/9.png;type=image/png' --output aa.png
