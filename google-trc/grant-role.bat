@echo off

@REM The project ID from TRC email that has access to free TPU quota
set PROJECT_ID=abc-123456
@REM The email you use to apply for TRC
set EMAIL=a@b.c

for %%r in (
    roles/iam.serviceAccountAdmin
    roles/resourcemanager.projectIamAdmin
    roles/tpu.admin
    roles/serviceusage.serviceUsageAdmin
    roles/iam.serviceAccountCreator
) do (
    echo %%r
    gcloud projects add-iam-policy-binding %PROJECT_ID% ^
        --member="user:%EMAIL%" ^
        --role="%%r" ^
        --condition=None
)


@REM Check current roles (format: yaml, json)
gcloud projects get-iam-policy %PROJECT_ID% --format=yaml > gcloud-roles.yaml
