Aliyun Build Docker
https://cr.console.aliyun.com/repository/cn-hangzhou/mario_odyssey/seed2023/build



## Verify
curl -X 'POST'   'http://localhost:8000/mask' \
 -H 'accept: application/json' \
 -H 'Content-Type: multipart/form-data' \
 -F 'file=@/Users/steven/my/code/2024SeedThyroid/datasets/origin/valid1/9.png;type=image/png' --output aa.png
