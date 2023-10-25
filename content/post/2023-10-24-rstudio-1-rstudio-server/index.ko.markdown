---
title: 코랩에서 Rstudio 사용하기 1. Rstudio-server 설치하기
author: JDW
date: '2023-10-24'
categories:
  - R
  - Colab
tags:
  - Colab
  - R
  - Rstudio
slug: rstudio-1-rstudio-server
lastmod: '2023-10-24T11:54:23+09:00'
featured: no
draft: no
image:
  caption: ''
  focal_point: ''
  preview_only: yes
---




![](images/colab_logo.png)

&nbsp;최근 <U>*나만의 딥러닝 워크스테이션 구성하기*</U> 프로젝트를 진행하면서 다방면으로 찾아보던 중 문득 **Colab**을 통해 `Rstudio`를 구동할 수 있지 않을까 싶었다. 그래서 찾아보니까 실제로 나와 같은 생각을 한 사람이 있었고, 직접 테스트를 진행해보니 성공적으로 Colab의 리소스로 `Rstudio`를 오픈 시킬 수 있어서 이를 소개하고자 한다. 

# Google Colaboratory 
&nbsp; **[Google Colab](https://colab.google/)**(이하 코랩)은 구글에서 제공하는 데이터 사이언스와 머신러닝을 위한 클라우드 기반 `Python` 개발 환경이다. 코랩의 장점은 여러가지 있지만 아무래도 **무료**로 `Jupyter notebook`을 사용할 수 있는 점을 꼽을 수 있겠다. 물론 free tier로 사용시에 리소스의 제한이 꽤나 엄격하지만 그래도 인터넷이 연결이 되고 `Python`을 다룰줄만 안다면 장소에 구애받지 않고 어디서든 작업을 할 수 있는 부분에 있어서는 유용한 서비스다. 

&nbsp; 코랩은 기본적으로 **Ubuntu** 기반의 가상머신에서 서버 리소스가 할당되어 사용되는 구조인듯 싶다. 코랩에서 사용되는 OS에 대한 기본 정보는 `Jupyter notebook`의 매직 명령어를 활용하여 확인할 수 있다.  

```bash
!cat /etc/*release
```

```
DISTRIB_ID=Ubuntu
DISTRIB_RELEASE=22.04
DISTRIB_CODENAME=jammy
DISTRIB_DESCRIPTION="Ubuntu 22.04.2 LTS"
PRETTY_NAME="Ubuntu 22.04.2 LTS"
NAME="Ubuntu"
VERSION_ID="22.04"
VERSION="22.04.2 LTS (Jammy Jellyfish)"
VERSION_CODENAME=jammy
ID=ubuntu
ID_LIKE=debian
HOME_URL="https://www.ubuntu.com/"
SUPPORT_URL="https://help.ubuntu.com/"
BUG_REPORT_URL="https://bugs.launchpad.net/ubuntu/"
PRIVACY_POLICY_URL="https://www.ubuntu.com/legal/terms-and-policies/privacy-policy"
UBUNTU_CODENAME=jammy
```

<br>

{{% callout note %}}

본격적으로 들어가기에 앞서 본 포스팅은 해당 [**github repository**](https://github.com/naru-T/RstudioServer_on_Colab)를 활용하여 작성되었음을 알립니다. 

{{% /callout %}}


# Rstudio-server 설치 
&nbsp; 기본적인 접근은 다음과 같다. 먼저 코랩 자체가 `Ubuntu`환경에서 실행되는 만큼 리눅스 우분투용 R과 Rstudio를 설치한다. 일반적으로 Ubuntu 환경 하에서는 로컬 웹 기반 `Rstudio-server`를 설치하여 활용하는데, 설치를 마치고 설치된 해당 주소와 포트를 터널링을 통해 접근하는 방식으로 진행된다. 각자 코랩을 켜서 아래의 **code-chunk**를 복사하여 셀 바이 셀로 실행시켜 나가면 된다. 

## 1. **rstudio** 유저 추가 

```bash
!sudo useradd --create-home --user-group --password $(echo '1234' | openssl passwd -1 -stdin) rstudio
!usermod -aG sudo rstudio
!su rstudio -c 'echo 1234 | sudo -S chmod -R 777 /var/log'
```
&nbsp; 먼저 우분투 환경에서 **"rstudio"** 사용자를 생성하고 `sudo` 권한을 부여한다. 이는 `rstudio-server` 로그인에 필요한 과정이다. 
&nbsp; 마지막 세번째 줄에 있는 `!su rstudio -c 'echo 1234 | sudo -S chmod -R 777 /var/log'`코드는 지금은 직접적인 연관이나 필요는 없지만 추후에 진행할 `tensorflow-R`을 설치할때 필요로 하기 때문에 미리 처리를 해 놓음.

&nbsp; 비밀번호의 경우 현재 *1234*로 세팅을 해 놓았는데 혹시라도도 `rstudio-server` 로그인 비밀번호를 바꾸고 싶다하면 코드에 있는 *1234*를 원하시는 비밀번호로 적어서 실행시키면 된다. 

## 2. 필수 프로그램 설치 

### GIS software 

```bash
%%capture # 출력 생략 
!apt-get update
!apt-get install --yes git ssh python3-venv
!apt-get install r-base r-base-dev gdal-bin python-gdal python3-gdal libgdal-dev libproj-dev proj-data proj-bin libgeos-dev libudunits2-dev libv8-dev libprotobuf-dev libxml2 libjq-dev
!apt-get install qgis saga
```

- `R` 설치와 더불어 `python3` 가상환경 모듈 및 지리 정보 및 공간 데이터 시각화를 위한 연계 패키지들을 설치

### Github CLI 

```bash
%%capture
!apt-key adv --keyserver keyserver.ubuntu.com --recv-key C99B11DEB97541F0
!echo | sudo apt-add-repository https://cli.github.com/packages
!apt update
!apt install gh
```

- `Github` 사용을 위한 `Github CLI(명령줄 인터페이스)` 패키지 설치 

### rstudio server

```bash
%%capture
!apt-get install gdebi-core
!wget https://download2.rstudio.org/server/focal/amd64/rstudio-server-2023.09.1-494-amd64.deb
!sudo gdebi rstudio-server-2023.09.1-494-amd64.deb
```

- `rstudio-server` 설치 진행 
- 업데이트 시 배포되는 `rstudio-server` 버전이 바뀌므로 최신 버전으로 사용을 원한다면 [https://posit.co/download/rstudio-server/](https://posit.co/download/rstudio-server/) 에서 최신 버전의 코드로 바꿔서 실행


<img src="images/install-rstudio.png" alt="rstudio-server posit 다운로드" width="50%"/>

<center>

*<rstudio-server posit 다운로드 페이지>*

</center>

### ngrok 토큰 발급 
&nbsp; 다음으로 진행해야 할 것으로 `ngork` 서비스 관련 사항들이다. `ngork`는 로컬의 개발환경을 인터넷으로 공개적으로 접근 가능하게 만들어주는 터널링(Tunneling) 서비스라고 한다. 주로 시스템 혹은 웹 개발시 테스트의 용도나 디버깅 작업을 수행하는데 사용되는 서비스인데 우리는 이것을 통해 코랩 내부의 `rstudio-server` 로컬 주소 및 포트를 접근할 수 있는 주소를 만드는데 사용할 것. [https://dashboard.ngrok.com/auth/your-authtoken](https://dashboard.ngrok.com/auth/your-authtoken) 다음에 사이트에서 ngork 가입 및 토큰을 발급을 진행하면 된다. 

<img src="images/ngork_auth.png" alt="ngork account 토큰 발급" width="50%"/>

<center>

*<ngork account 토큰 발급>*

</center>


```python
from getpass import getpass

# Don't forget create your account of ngrok and get token from https://dashboard.ngrok.com/auth/your-authtoken
authtoken = getpass("Input your Auth token")
```

- 발급받은 토큰을 복사하여 아래의 코드 실행 시 생성된 입력 창에 입력  


```bash
! wget -q -c -nc https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
! unzip -qq -n ngrok-stable-linux-amd64.zip
```

- `wget`과 `unzip` 명령을 통해 `ngrok`을 다운로드하고 압축을 해제하여 코랩 환경에서 IP 터널링을 가능케 함

## 3. ngrok을 통한 Rstudio server 가동 

&nbsp; 마지막으로 앞서 설정한 `ngork` 토큰을 통해 코랩 로컬에서 실행되어 있는 `rstudio-server` 웹 서버를 공용 IP 주소로 터널링하여 접근을 진행한다. 아래의 코드를 실행시키면 나오는 주소가 바로 코랩으로 실행시킨 `rstudio-server`의 웹 주소이므로 출력하여 나오는 주소로 접속.


```python
# Run ngrok
get_ipython().system_raw('./ngrok authtoken $authtoken && ./ngrok http 8787 &')
! sleep 3

# Get the address for Rstudio-server
import requests
from re import sub
r = requests.get('http://localhost:4040/api/tunnels')

str_ssh = r.json()['tunnels'][0]['public_url']
print(str_ssh)
```

```
#> https://af9e-105-199-235-17.ngrok-free.app # 출력 예시
```

<img src="images/ngork_visit.png" width="100%" /><img src="images/rstudio-login.png" width="100%" />

<img src="images/rstudio-main.png" alt="" width="100%"/>

<center>

*<ngork tunneling 및 Rstudio-server login, main 페이지>*

</center>

&nbsp; 출력으로 나온 `ngork` 터널링 페이지에서 **Visit Site**를 클릭하면 우리가 설정한 `rstudio-server`의 로그인 화면을 볼 수 있고, 이 후에 앞서 설정한 로그인 입력정보`(ID : rstudio, PW : 1234)`를 입력하게 되면 비로소 코랩에서 코랩의 리소스를 통한 `Rstudio` 사용 가능하다.

------------------------------------------------------------------------

# References 

- [https://github.com/naru-T/RstudioServer_on_Colab](https://github.com/naru-T/RstudioServer_on_Colab) (Github repository) 

- [https://github.com/naru-T/RstudioServer_on_Colab/blob/master/Rstudio_server.ipynb](https://github.com/naru-T/RstudioServer_on_Colab/blob/master/Rstudio_server.ipynb) (.ipynb) 










