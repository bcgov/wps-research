# How To: 

## Install AWS CLI from Git Bash

Contributing to this repository requires the installation of Amazon Web Service's \(AWS\) command line interface \(CLI\). The instructions for installing the aws cli can be found in **<u>[Amazon's documentation](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)</u>**, which is a straightforward resource for accomplishing this task, regardless of your operating system.

Here's an <u>example</u> of how to complete the installation through a 64-bit Windows OS:

1. Download and run the AWS CLI MSI installer for Windows (64-bit): [`https://awscli.amazonaws.com/AWSCLIV2.msi`](https://awscli.amazonaws.com/AWSCLIV2.msi) 


    *Alternative:*

    You can also run the `msiexec`` command to run the MSI installer: 

    ```
    msiexec.exe /i https://awscli.amazonaws.com/AWSCLIV2.msi
    ```

2. To confirm the installation, open the Start menu, search for `cmd` to open a command prompt window, and at the command prompt type in the following:

```
aws --version
```

If Windows is unable to find the program, you might need to close and reopen the command prompt window to refresh the path. Another option is to follow the troubleshooting in [Troubleshoot AWS CLI errors](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-troubleshooting.html).