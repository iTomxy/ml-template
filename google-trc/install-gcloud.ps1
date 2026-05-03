# install-gcloud.ps1
# FROM: https://docs.cloud.google.com/sdk/docs/install-sdk
# RUN: powershell.exe -ExecutionPolicy Bypass -File install-gcloud.ps1

(New-Object Net.WebClient).DownloadFile("https://dl.google.com/dl/cloudsdk/channels/rapid/GoogleCloudSDKInstaller.exe", "$env:Temp\GoogleCloudSDKInstaller.exe")

& $env:Temp\GoogleCloudSDKInstaller.exe
