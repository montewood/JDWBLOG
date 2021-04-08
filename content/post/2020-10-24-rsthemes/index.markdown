---
title: '[R] Rstudio의 테마를 변경해 사용하기(With rsthemes Package)'
author: JDW
date: '2020-10-24'
slug: rsthemes
categories:
  - R
tags:
  - R
  - Rstudio
  - rsthemes
subtitle: ''
summary: ''
authors: []
lastmod: '2021-04-08T17:58:08+09:00'
featured: no
image:
  caption: ''
  focal_point: ''
  preview_only: yes
projects: []
output:
  blogdown::html_page:
    toc: TRUE
    toc_depth: 4
---


<img src="https://media.giphy.com/media/Ai6o3zm3NwvOazPxkm/giphy.gif" alt="" width="100%"/>

&nbsp;데이터분석 혹은 R프로그래밍을 진행할 때 가장 많이 보는것이 (아마도) 구글, 스택오버플로우, 그리고 Rstudio일 것입니다.대다수의 R 유저들은 사용하는 IDE로서 **Rstudio** 를 선호하는데요. 저 역시도 Rstudio에서 제공하는 각종 편의기능에 익숙해져 vscode와 같은 다른 IDE로 쉽사리 넘어가지 못하는것 같습니다. 그러나 오랜시간 비슷한 화면만을 보다보면 제아무리 만족스럽던 것도 점차 지루해지기 쉽상입니다. 이럴때 약간의 변화만 주어도 한결 나은 코딩을 할 수 있을텐데요. 오늘은 Rstudio에서 제공하는 기본테마를 벗어나 새롭고 다양한 테마를 제공할 뿐만아니라 테마변경을 손쉽게 하는 [__`rsthemes`__](https://github.com/gadenbuie/rsthemes) 패키지를 소개하고자 합니다.  

# 기본 테마 변경   
&nbsp;Rstudio를 처음 설치하면 4개의 패널로 나뉜 흰색의 기본 테마를 마주치게 됩니다. 기본 테마 역시 멋지지만 오랜시간 흰색의 배경을 응시하다보면 눈이 쉽사리 피로해지곤 하기때문에 많은 분들이 개발환경에 있어서 밝은것보다 어두운 테마를 더욱 선호하는것 같습니다. 먼저 기본적인 Rstudio의 테마의 변경은 상단의 Tools > Global Options > Appearence 탭을 통해 변경이 가능합니다. 

![](images/K-20201011-162316.png)

![](images/K-20201011-162354.png)

![](images/K-20201011-162403.png)
<center><em>기본 테마 변경 과정</em></center>

# rsthemes package

&nbsp;기본적으로 제공되는 테마가 불충분하다고 느껴지신다면 이 패키지가 도움이 될 수도 있습니다. 서두에서도 소개했다시피 [**`rsthemes`**](https://github.com/gadenbuie/rsthemes)패키지는 Rstudio에서 사용할 수 있는 90여가지의 추가적인 테마를 제공합니다.

## 설치 
&nbsp;[**`rsthemes`**](https://github.com/gadenbuie/rsthemes)는 cran에 정식으로 등록되지 않은 패키지이기 때문에 개발자 깃헙 레포지터리에서 다운받을 수 있습니다. 

```r
# install.packages("devtools")
devtools::install_github("gadenbuie/rsthemes")
```
&nbsp;패키지를 다운받은 후, 추가 테마를 설치합니다. 

```r
rsthemes::install_rsthemes()
```
&nbsp; base16테마를 설치하는 옵션을 지정하면 더욱 많은 테마를 설치할 수 있습니다.  

```r
rsthemes::install_rsthemes(include_base16 = TRUE)
```

## 사용
&nbsp;패키지와 테마의 설치까지 마치셨다면 [**`rsthemes`**](https://github.com/gadenbuie/rsthemes)에서 제공하는 함수들을 통해 테마를 관리 할 수 있습니다. 
&nbsp;맨 먼저 설치된 테마의 목록들을 확인합니다. 

```r
# 설치된 테마의 목록 확인
rsthemes::list_rsthemes()
#  [1] "a11y-dark {rsthemes}"                   
#  [2] "a11y-light {rsthemes}"                  
#  [3] "base16 3024 {rsthemes}"                 
#  [4] "base16 Apathy {rsthemes}"               
#  [5] "base16 Ashes {rsthemes}"                
#  [6] "base16 Atelier Cave {rsthemes}"         
#  [7] "base16 Atelier Dune {rsthemes}"         
#  [8] "base16 Atelier Estuary {rsthemes}"      
#  [9] "base16 Atelier Forest {rsthemes}"       
# [10] "base16 Atelier Heath {rsthemes}"        
# [11] "base16 Atelier Lakeside {rsthemes}"     
# [12] "base16 Atelier Plateau {rsthemes}"      
# [13] "base16 Atelier Savanna {rsthemes}"      
# [14] "base16 Atelier Seaside {rsthemes}"      
# [15] "base16 Atelier Sulphurpool {rsthemes}"  
# [16] "base16 Bespin {rsthemes}"               
# [17] "base16 Brewer {rsthemes}"               
# [18] "base16 Bright {rsthemes}"               
# [19] "base16 Chalk {rsthemes}"                
# [20] "base16 Codeschool {rsthemes}"           
# [21] "base16 Cupcake {rsthemes}"              
# [22] "base16 Darktooth {rsthemes}"            
# [23] "base16 Default Dark {rsthemes}"         
# [24] "base16 Default Light {rsthemes}"        
# [25] "base16 Dracula {rsthemes}"              
# [26] "base16 Eighties {rsthemes}"             
# [27] "base16 Embers {rsthemes}"               
# [28] "base16 Flat {rsthemes}"                 
# [29] "base16 Google Dark {rsthemes}"          
# [30] "base16 Google Light {rsthemes}"         
# [31] "base16 Grayscale Dark {rsthemes}"       
# [32] "base16 Grayscale Light {rsthemes}"      
# [33] "base16 Green Screen {rsthemes}"         
# [34] "base16 Gruvbox dark, hard {rsthemes}"   
# [35] "base16 Gruvbox dark, medium {rsthemes}" 
# [36] "base16 Gruvbox dark, pale {rsthemes}"   
# [37] "base16 Gruvbox dark, soft {rsthemes}"   
# [38] "base16 Gruvbox light, hard {rsthemes}"  
# [39] "base16 Gruvbox light, medium {rsthemes}"
# [40] "base16 Gruvbox light, soft {rsthemes}"  
# [41] "base16 Harmonic16 Dark {rsthemes}"      
# [42] "base16 Harmonic16 Light {rsthemes}"     
# [43] "base16 Hopscotch {rsthemes}"            
# [44] "base16 IR Black {rsthemes}"             
# [45] "base16 Isotope {rsthemes}"              
# [46] "base16 London Tube {rsthemes}"          
# [47] "base16 Macintosh {rsthemes}"            
# [48] "base16 Marrakesh {rsthemes}"            
# [49] "base16 Materia {rsthemes}"              
# [50] "base16 Mexico Light {rsthemes}"         
# [51] "base16 Mocha {rsthemes}"                
# [52] "base16 Monokai {rsthemes}"              
# [53] "base16 Nord {rsthemes}"                 
# [54] "base16 Ocean {rsthemes}"                
# [55] "base16 OceanicNext {rsthemes}"          
# [56] "base16 OneDark {rsthemes}"              
# [57] "base16 Paraiso {rsthemes}"              
# [58] "base16 PhD {rsthemes}"                  
# [59] "base16 Pico {rsthemes}"                 
# [60] "base16 Pop {rsthemes}"                  
# [61] "base16 Railscasts {rsthemes}"           
# [62] "base16 Rebecca {rsthemes}"              
# [63] "base16 Seti UI {rsthemes}"              
# [64] "base16 Shapeshifter {rsthemes}"         
# [65] "base16 Solar Flare {rsthemes}"          
# [66] "base16 Solarized Dark {rsthemes}"       
# [67] "base16 Solarized Light {rsthemes}"      
# [68] "base16 Spacemacs {rsthemes}"            
# [69] "base16 Summerfruit Dark {rsthemes}"     
# [70] "base16 Summerfruit Light {rsthemes}"    
# [71] "base16 Tomorrow Night {rsthemes}"       
# [72] "base16 Tomorrow {rsthemes}"             
# [73] "base16 Twilight {rsthemes}"             
# [74] "base16 Unikitty Dark {rsthemes}"        
# [75] "base16 Unikitty Light {rsthemes}"       
# [76] "base16 Woodland {rsthemes}"             
# [77] "Fairyfloss {rsthemes}"                  
# [78] "Flat White {rsthemes}"                  
# [79] "GitHub {rsthemes}"                      
# [80] "Nord Polar Night Aurora {rsthemes}"     
# [81] "Nord Snow Storm {rsthemes}"             
# [82] "Oceanic Plus {rsthemes}"                
# [83] "One Dark {rsthemes}"                    
# [84] "One Light {rsthemes}"                   
# [85] "Solarized Dark {rsthemes}"              
# [86] "Solarized Light {rsthemes}"      
```
&nbsp;다음의 함수를 통해 설치된 테마들을 하나씩 적용 할 수 있습니다. 

```r
# 테마 적용하기 
rsthemes::try_rsthemes()
# light, dark, base16 themes를 선택적으로 적용하기 
rsthemes::try_rsthemes("light")
```

&nbsp;`rstudioapi::applyTheme()`를 사용하면 목록의 테마를 하나씩 적용할 수 있습니다. 

```r
# 특정 테마 적용하기 
rstudioapi::applyTheme("One Dark {rsthemes}")
```

## 손 쉽게 테마 변경하기 
&nbsp;[**`rsthemes`**](https://github.com/gadenbuie/rsthemes)패키지가 가진 가장 큰 강점이 여기에 있습니다. 기호에 맞는 테마를 저장하고, 이를 단축키로 등록시켜 손 쉽게 테마를 전환시킬 수 있는 기능을 지원하는데요. 이 기능을 사용하기 위해선 먼저 자신이 선호하는 테마를 등록해야 합니다. 

### 주-야간 테마
&nbsp;`rsthemes::set_theme_light()`, `rsthemes::set_theme_dark()`함수는 각각 밝은 테마, 어두운 테마 하나씩을 지정할 수 있습니다. 

```r
# 현재 사용중인 테마를 주-야간 테마로 설정하기 
rsthemes::set_theme_light()
rsthemes::set_theme_dark()

# 특정 테마를 주-야간 테마로 등록하기 
rsthemes::set_theme_light("One Light {rsthemes}")
rsthemes::set_theme_dark("One Dark {rsthemes}")
```

### 즐겨찾는 테마
&nbsp;`rsthemes::set_theme_favorite()`함수를 통해 자신이 선호하는 테마 등록이 가능합니다. 

```r
# 현재 사용중인 테마를 선호테마목록에 추가 
rsthemes::set_theme_favorite()

# 다수의 테마 선호테마목록에 추가 
rsthemes::set_theme_favorite(
  c("GitHub {rsthemes}", "One Light {rsthemes}", "One Dark {rsthemes}")
)
```

### 기본 세션에 등록 
&nbsp;위의 함수들을 통해 선호하는 테마를 지정하였습니다. 하지만 이 설정은 오직 현재 세션에만 저장이 되기 때문에 만약 Rstudio를 종료 후 다시 키거나 혹은 새로운 세션으로 접속하면 사라질 정보입니다. 때문에 기본 세션에 등록을 시켜야지만 지속적으로 사용이 가능합니다.  `usethis::edit_r_profile()`함수를 통해 `~/.Rprofile`을 편집하여 등록한 테마를 모든 세션에서 사용 가능하도록 설정합니다. 

```r
# ~/.Rprofile
if (interactive()) {
  rsthemes::set_theme_light("GitHub {rsthemes}")
  rsthemes::set_theme_dark("Fairyfloss {rsthemes}")
  rsthemes::set_theme_favorite(
    c("GitHub {rsthemes}", 
      "One Light {rsthemes}", 
      "One Dark {rsthemes}")
  )
}
```

### 단축키 등록 
&nbsp;`~/.Rprofile`파일에 테마를 등록하여 저장하면 세션이 종료되거나 바뀌어도 설정은 날아가지 않습니다. 이번엔 저장된 테마를 스위칭하는 단축키를 설정하는 방법에 대해 알아보겠습니다. [**`rsthemes`**](https://github.com/gadenbuie/rsthemes)패키지는 세가지 addins 기능을 지원합니다. <br>

- **Toggle Dark Mode**  : 주-야간 테마 
- **Favorite Themes**   : 즐겨찾는 테마 
- **Auto Dark Mode**    : 주-야간 테마 지정된 시간에 자동 변경 

&nbsp; addins에 들어가있는 기능들은 기본적으로 Rstudio의 단축키로 설정이 가능합니다. 이 말인 즉슨 테마전환기능을 단축키로 지정해서 사용하는게 가능하다는 의미인데요. Rstudio의 단축키는 Tools > Modify Keyboard shortcuts에서 설정 할 수 있습니다.  

![](images/K-001-1.png)

![](images/K-20201013-102841.png)

&nbsp;검색란에 'toggle'만 타이핑해도 "Toggle Dark Mode"이 자동완성으로 등장합니다. 이곳에 원하는 단축키를 등록합니다. 저는 단축키로 "Ctrl+Alt+D"를 등록하였습니다. 

![](images/K-20201013-104228.png)

![](images/K-20201013-104815.png)

&nbsp;이후 설정한 단축키를 누르면 주-야간테마로 전환되는 것을 확인할 수 있을 겁니다. 즐겨찾는 테마의 단축키 등록도 이와 동일한 방식으로 등록이 가능하므로 똑같은 방식이니 추가적인 설명은 생략하고 넘어가도록 하겠습니다. (단축키로 "Ctrl+Alt+N"을 추천합니다.)

&nbsp;addins의 마지막 기능인 **Auto Dark Mode**를 활성화하면  Rstudio가 정해진 시간에 맞춰서 주간테마, 야간테마를 자동으로 전환됩니다 직접 테마를 바꾸거나 단축키 등록을 하지 않더라도 이것만 설정하면 낮과 밤에 따라 테마를 변경시킬 수 있는데요. 이 기능은 단축키 등록이 아닌 아까 살펴봤던 `~/.Rprofile`의 수정을 통해 가능합니다. 콘솔창에 다시 한번 `usethis::edit_r_profile()`함수를 입력합니다. 


```r
# ~/.Rprofile
if (interactive()) {
  rsthemes::set_theme_light("GitHub {rsthemes}")       # 주간 테마  
  rsthemes::set_theme_dark("Fairyfloss {rsthemes}")    # 야간 테마  
  rsthemes::set_theme_favorite(                        # 즐겨찾는 테마  
    c("GitHub {rsthemes}", 
      "One Light {rsthemes}", 
      "One Dark {rsthemes}")
  )
  
  # 주-야간 테마 지정된 시간에 자동 변경
  setHook("rstudio.sessionInit", function(isNewSession) {
    rsthemes::use_theme_auto(dark_start = as.POSIXct('17:00',format="%H:%M"), dark_end = as.POSIXct('09:00',format="%H:%M"))
  }, action = "append")
}
```

&nbsp;`~/.Rprofile`에 *'주-야간 테마 지정된 시간에 자동 변경'* 코드를 추가하였습니다. 만약 이와 같이 설정하면 17:00에 자동으로 야간테마로 전환되고, 다음날 09:00가 되면 주간 테마로 전환될 것입니다. 함수 내부에 `as.POSIXct()`값을 조정하여 테마가 전환되는 시간을 조정할 수 있습니다. 

___

# 참고자료 

- [Garrick Aden‑Buie Blog (rsthemes)](https://www.garrickadenbuie.com/project/rsthemes/)

