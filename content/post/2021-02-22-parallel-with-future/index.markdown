---
title: '[R] 더욱 빠른 연산을 위한 병렬처리(With future Package)'
author: JDW
date: '2021-02-22'
slug: parallel-with-future
categories:
  - R
tags:
  - R
  - 비동기
  - Async
  - 병렬
  - parallel
  - future
  - 속도
subtitle: ''
summary: ''
authors: []
lastmod: '2021-02-22T18:34:01+09:00'
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




<img src="images/main.jpg" alt="" width="70%"/>

<center>

*사진출처 :* <a href="https://pixabay.com/users/mohamed_hassan-5229782/?utm_source=link-attribution&amp;utm_medium=referral&amp;utm_campaign=image&amp;utm_content=3042638">mohamed Hassan</a> from <a href="https://pixabay.com/?utm_source=link-attribution&amp;utm_medium=referral&amp;utm_campaign=image&amp;utm_content=3042638">Pixabay</a>

</center>

<br>

 계산할게 많은 코드를 돌릴때면 당연하게도 많은 시간을 소모하게 됩니다. 통계모델구축과 같이 연산량이 많은 작업을 돌리는 경우에 종종 이러한 상황을 맞닥들이곤 하는데요. 코드가 돌아가는 시간동안은 어쩔수 없이 하고있는 업무를 잠시 중단하거나 다른 일을 하는 수 밖에 없는 상황입니다. 이와 같은 시간이 아깝다고 생각이 드는건 지극히 정상적인 생각일 것입니다. 코드의 결과물이 빨리 나올수록 하고 있는 일을 더 빨리 끝내거나 혹은 그 시간에 더 많은 일을 할 수 있는데 말이죠. 오늘은 이러한 코드의 연산에 필요한 시간을 어쩌면 획기적으로 줄여줄수 있는 **`future`** 패키지에 대해 알아보겠습니다.

# CPU

 설명하기에 앞서 여려분은 R을 어떠한 환경에서 사용하고 계신가요? 노트북 또는 데스크탑, 클라우드 환경 등 다양한 환경에서 R을 사용하고 계실겁니다. 저는 윈도우가 탑제된 노트북에서 R을 사용하고 있는데요. 제가 사용하고있는 노트북의 CPU는[*인텔® 코어™ i7-8750H 프로세서*](https://ark.intel.com/content/www/kr/ko/ark/products/134906/intel-core-i7-8750h-processor-9m-cache-up-to-4-10-ghz.html)로 헥사코어(6개의 코어)와 인텔의 하이퍼쓰레딩 기술이 적용된 제품을 사용하고 있습니다. CPU는 들어봤어도 헥사코어니 하이퍼쓰레딩이니하는 용어에 대해서 처음 들어보시는 분들도 계실겁니다. 저와 같은 *컴알못*(컴퓨터를 잘 알지 못하는)분들에게는 이런 생소한 용어 하나하나가 어려움으로 다가올텐데요. 이 후에 후술할 **병렬처리**를 이해하기 위해서는 이러한 부품들의 역할과 기능을 이해하고 있어야 하기 때문에 이런 용어를 설명해야 하는점 미리 양해를 구하겠습니다. 혹시나 이와같은 개념을 이미 알고 계신다면 넘어가셔도 상관 없습니다.

## 코어 & 쓰레드

 용어에 대한 설명을 진행하겠지만 본 포스트는 컴퓨터 용어에 대한 것이 아니기 때문에 최대한 간단하게 해보도록 하겠습니다. 먼저 컴퓨터의 두뇌라 할수있는 CPU에는 코어라는 것이 존재하는데 코어는 CPU에서 실질적으로 연산을 담당하는 장치입니다. 과거에는 CPU당 1개의 코어가 들어갔었으나 기술이 발전하면서 현재 출시되는 CPU에는 많게는 코어가 32개까지 들어가는 *멀티코어*로 구성되어 있습니다.

 이러한 코어에는 쓰레드라는 개념이 존재하는데, 쓰레드는 코어에 할당된 작업 공간입니다. 즉, CPU 내부에서 일을 처리하는 최소의 단위이며 코어가 일을 하는 노동자라고 하면 쓰레드는 노동자가 동시에 할 수 있는 업무의 갯수라고 이해하면 될 것 같습니다.

<img src="images/cpu_core.png" alt="" width="65%"/>

<center>

*CPU 내부 코어의 구성*

</center>

<br>

 예시로 든 이미지에 있는 CPU는 4개의 코어가 들어간 *쿼드코어* CPU로서 당연한 말이지만 2개의 코어를 지닌 *듀얼코어*나 *싱글코어*의 CPU보다 높은 성능을 낼 것입니다. 한편 인텔에는 *하이퍼쓰레딩*이라는 기술이 존재하는데 일반적으로 CPU의 코어는 코어당 하나의 쓰래드를 배분받아 일을 하게 되지만 해당 기술은 하나의 코어가 두 개의 쓰레드를 할당받아 마치 2개의 코어처럼 작동을 하는 것을 의미합니다. 즉, 헥사코어에 하이퍼쓰레드 기술을 지니고 있는 저의 노트북같은 경우는 동시에 연산처리를 할 수 있는 실질적인 코어가 총 12개가 있는 것이죠.

<img src="images/cpuz.png" alt="" width="50%"/>

<center>

*"CPU-Z" 프로그램으로 알아본 CPU 정보 및 코어 & 쓰레드 갯수*

</center>

<br>

# 병렬처리와 비동기 프로그래밍

 한가지 상황을 상상해보죠. 오늘 저녁은 직접 요리를 해서 먹기로 마음 먹었습니다. 오늘은 집에서 갈비찜에 시금치 무침, 잡채, 그리고 김치볶음밥을 하기로 정했죠. 요리를 위한 재료구입과 다듬기를 마친 상황이고, 이제 한참 갈비찜을 만들고 있었습니다. 그런데 생각보다 갈비찜을 만드는데 시간이 많이 소요가 됩니다. 양념장에 재운 갈비를 끓여 익히기만 하면 되는데 그 시간이 한시간정도 걸린다고 합니다. 이럴경우 보통 어떻게 해야 할까요? 당연하게도 갈비를 익히는 시간동안 다른 잡채나 시금치 무침을 하면 됩니다.

 그러던 와중에 집에서 요리한다는 소식을 접한 친구가 갑작스럽게 우리집에 방문한다고 합니다. 잘됐습니다. 공짜로 얻어먹으려는 친구를 부려서 같이 요리 하면 좋겠네요. 아마도 시간이 훨씬 단축될 것입니다. 원래라면 두 시간 걸릴 요리가 친구가 도와줌으로 인해서 한 시간 정도로 단축 될 것 같네요.

 혼자 처리할 일을 친구와 같이 함으로서 시간이 단축되는 것 처럼 단일 프로세스에서 처리할 일을 여러 프로세스와 함께 처리하는 방식을 병렬처리라고 합니다. 그리고 일반적인 R 코드는 순차적으로 처리를 하지 동시에 처리하는것은 불가능합니다. 이를 위의 상황에 비유하자면 갈비찜과 잡채를 만들고 있는 상황에서 갈비찜이 완성되기 전까지는 잡채나 시금치 무침같은 다른 요리를 만드는 것에는 일절 손도 안대는 상황이라 할 수 있습니다. 이처럼 작업의 순서가 고정되어 있지 않고 주어진 일을 여건에 맞게 유기적으로 처리하는 방식을 비동기(Async) 방식이라 하며, 반대로 주어진 일을 순차적으로 처리하는 방식을 동기식(Sync)이라 합니다.

# future package

 동기식으로 처리되던 기존의 R 코드를 **`future`** 패키지를 사용한다면 효율적이고 그리고 손쉽게 병렬 처리 및 비동기 연산을 할 수 있습니다.

## 설치 및 기본 설정

 **`future`** 패키지는 cran에 정식으로 등록된 패키지이기 때문에 `install.packages()`함수를 사용하여 설치가 가능합니다.


```r
# install.packages(future)
library(future)
```

 패키지를 설치하였다면 비동기 연산을 위한 설정을 별도로 진행해 줘야 합니다. **`future`** 패키지가 제공하는 `plan()` 함수는 코드 연산 방식을 동기식으로 할지 비동기식으로 할지 설정하도록 하는 함수입니다. 여기에 인자로 `multisession` 을 지정하면 **`future`** 패키지가 백그라운드에서 자동적으로 별도의 R 세션을 생성하고, 이후 병렬연산이 필요한 경우에 사용할 수 있습니다. (MAC 이용자의 경우 `plan(multicore)`로 설정하시면 됩니다.)


```r
plan(multisession)
```

 `plan()` 함수를 실행하면 현재 환경에서 사용 가능한 전체 CPU의 코어를 자동으로 탐지하여 이 갯수만큼의 R 세션을 생성하게 됩니다. 저의 노트북을 예시로 들면 12개의 세션을 백그라운드에서 생성하게 되죠. 만약 전체 CPU의 코어 사용이 부담스럽다면 `plan()` 함수가 제공하는 `workers` 옵션을 설정하면 원하는 만큼의 코어를 사용할 수 있습니다. 여기서는 8개의 코어만 사용해 보도록 하겠습니다.


```r
# 기본 설정 코어 갯수 & 실행 가능한 최대 CPU 코어
availableCores()
```

```
#> system 
#>     12
```

```r
# 사용 코어 갯수 지정
plan(multisession, workers = 8)
```

 이렇게 설정되어 만들어진 세션들은 (윈도우)작업관리자를 통해 확인할 수 있습니다. `plan()`함수를 통해 'R for Windows front-end' 세션들이 새로이 생성된 것을 볼 수 있습니다.

<img src="images/future-newsession.png" alt="" width="100%"/>

<center>

*기본 세션(좌), `plan(multisession)`으로 생성된 백그라운드 세션(우)*

</center>

<br>

## 기본 용법

 **`future`** 패키지를 사용하는 방법에는 두가지 방식이 있습니다. 하나는 `future()` 함수를 명시적으로 사용하는 것이고 다른 하나는 패키지에서 제공하는 `%<-%` 형태의 특수한 연산자를 사용하여 퓨쳐를 암시적으로 지정하는 것입니다.\
 비교를 위해 임의의 함수 하나를 생성해서 사용해보도록 하겠습니다. `testFunc1()` 함수는 단순히 'hi'라는 단어를 콘솔창에 출력한 뒤 3.14를 반환하는 함수입니다.


```r
testFunc1 <- function(){
  cat('hi')
  3.14
}
```

 그런다음 **`future`** 의 명시적 용법과 암시적 용법을 각각 사용하여 `testFunc1()` 함수를 출력해보도록 하겠습니다. 두 방식 모두 이전에 설정한 `plan(multisession)` 함수를 통해 비동기 방식을 설정한 상태이며, 이를 통해 연산이 현재의 R studio 세션에서 진행되는 것이 아닌 백그라운드 세션에서 이루어집니다.


```r
# 명시적 용법 
f1 <- future({
  testFunc1()
})

# 암시적 용법 
f2 %<-% {
  testFunc1()
}
```


```r
f1
```

```
#> MultisessionFuture:
#> Label: '<none>'
#> Expression:
#> {
#>     testFunc1()
#> }
#> Lazy evaluation: FALSE
#> Asynchronous evaluation: TRUE
#> Local evaluation: TRUE
#> Environment: R_GlobalEnv
#> Capture standard output: TRUE
#> Capture condition classes: 'condition'
#> Globals: 1 objects totaling 5.21 KiB (function 'testFunc1' of 5.21 KiB)
#> Packages: <none>
#> L'Ecuyer-CMRG RNG seed: <none> (seed = FALSE)
#> Resolved: TRUE
#> Value: <not collected>
#> Conditions captured: <none>
#> Early signaling: FALSE
#> Owner process: 81d55a71-a4c6-da64-cea2-9ca8570374dc
#> Class: 'MultisessionFuture', 'ClusterFuture', 'MultiprocessFuture', 'Future', 'environment'
```


```r
f2
```

```
#> hi
```

```
#> [1] 3.14
```

 암시적 용법을 사용한 `f2` 의 경우 `testFunc1()` 함수의 output을 정상적으로 출력하였지만 명시적 용법을 사용한 `f1` 객체의 경우는 다소 알기 어려운 메세지만을 출력하였습니다. **`future`** 패키지에서 명시적 용법과 암시적 용법은 기본적으로 같은 기능을 수행하지만 명시적 용법을 사용하는 경우 값을 반환받기 위해서는 `value()` 함수를 통해 반환을 받습니다.


```r
value(f1)
```

```
#> hi
```

```
#> [1] 3.14
```

 `value()` 함수를 사용하니 `f1` 객체 역시 `f2` 와 동일한 값을 반환합니다. 이처럼 명시적 용법을 사용한 퓨쳐에서는 값을 반환받기 위해 `value()` 함수를 사용해야 하며, 간결하게 사용하고 싶은 경우에는 암시적 용법인 `%<-%` 연산자를 사용하면 됩니다.

## 활용

  지금까지 **`future`** 패키지의 설치와 기본적인 사용법에 대해서 알아보았습니다. 그렇다면 어떠한 상황 혹은 어떠한 목적을 위하여 이것을 사용해야 할까요? 여러가지 상황이 존재하겠지만 제가 생각하기에 **`future`** 패키지를 알맞게 사용하는 상황은 크게 두 가지 인 것 같습니다.

### 1) 연산을 다른 세션에 위임할 때

 맨 처음 말한것처럼 연산량이 많은 코드를 돌릴 경우 코드가 온전히 돌아갈때 까지 기다려야 하는 상황이 발생합니다. 갈비찜이 완성되기 전까지는 아무런 동작도 하지 않는셈이죠. 이럴때 **`future`** 패키지를 사용하면 현재 작업하고 있는 세션에 방해되지 않게 코드를 돌릴 수가 있습니다.


```r
testFunc2 <- function(x){
  Sys.sleep(x)
  return(print('Done!'))
}
```

 시범을 위해 역시 임의의 함수를 생성하도록 하겠습니다. `testFunc2()` 함수는 지정한 `x`초 만큼 동작을 멈추었다가 마지막에 `print('Done!')`을 수행하는 함수입니다. 즉, `x`초간 *멈추는 동작을 수행* 한 뒤 'Done!'이란 글자를 출력하는 함수이죠. 이를 계산이 오래 걸리는 함수라고 가정하고 다른 세션에 위임하여 10초동안 돌려보도록 하겠습니다.


```r
# 다른 세션에서 연산하도록 위임 
t1 <- future({
  testFunc2(10)
})

# 결과 출력 
value(t1)
```

```
#> [1] "Done!"
```

 **`future`** 패키지를 통해 연산을 다른 세션에 위임한다는 사실을 알게 되었습니다. 그런데 다른 세션에서 작업이 완료가 되었는지 확인은 어떻게 해야할까요? 현재 세션에서 코드를 돌린다면 코드의 연산이 끝날 시 콘솔창에 즉각 출력이되어 코드의 수행 여부를 즉각 확인할 수 있을 것입니다. 그렇지만 다른 세션에서 돌아간 코드는 연산이 끝나더라도 현재 세션에 아무런 영향을 주지 않기 때문에 이를 확인할 방도가 없습니다.\
<br>  이러한 상황을 대비하여 **`future`** 패키지는 `resolved()` 함수를 제공하고 있습니다.


```r
# 다른 세션에 연산 위임 
t2 <- future({
  testFunc2(10)
})

# 연산 확인 
i = 0
while(T){
    cat(paste0(i, ' now running \n'))
    Sys.sleep(1)
    i = i + 1
    if(resolved(t2)){
        cat('Calculate done! \n')
        break
    }
}

# 결과 출력 
value(t1)
```

```
#> 0 now running 
#> 1 now running 
#> 2 now running 
#> 3 now running 
#> 4 now running 
#> 5 now running 
#> 6 now running 
#> 7 now running 
#> Calculate done! 
#> [1] "Done!"
```

 `t2` 객체가 현재 연산중인지를 확인하기 위하여 중간에 `while` 문을 추가하였습니다. `while` 문의 기능은 1초 간격으로 동작을 멈추었다가 "`i` now running"을 출력하는 코드인데 중간에 있는 `if` 문에 `resolved(t2)` 함수가 있는 것을 볼 수 있습니다. `resolved()` 함수의 기능은 백그라운드 세션에서 연산중인 퓨쳐가 동작하는지를 확인하기 위한 함수로서 만약 해당 퓨쳐가 연산중에 있으면 `FALSE` 를, 연산을 마쳤다면 `TRUE` 를 반환하게 됩니다. 따라서 `t2` 객체에 지정된 퓨쳐가 백그라운드 세션에서 연산중이면 "`i` now running" 출력과 1초 간격으로 동작을 멈추는 작동을 수행하고, 연산이 끝난경우 `if` 문 내부의 `break` 를 통해 `while` 문이 종료가 되는 것입니다. 이와 같은 방법을 응용한다면 백그라운드에서 실행중인 비동기 연산의 수행완료 여부를 효율적으로 인지할 수 있을 것입니다.

### 2) 더욱 빠른 연산이 필요로 할 때

#### 2-1) 파일 읽기 속도 비교

 두 번째 경우는 연산속도를 높이고 싶은 상황에 사용하는 것입니다. **`future`** 패키지가 지원하는 병렬연산을 적극 사용하는 방식인데 속도의 비교를 위해 테스트용 데이터를 생성해 보도록 하겠습니다.


```r
# 테스트용 csv 저장
library(dplyr)
data.frame(
    t1 = rnorm(10000),
    t2 = rnorm(10000),
    t3 = rnorm(10000)
) %>% write.csv(., 'testFile.csv')
```

  정규분포를 따르는 난수를 생성하는 `rnorm()` 함수를 사용하여 3개의 column과 10000개의 row를 지니고 있는 임의의 데이터프레임을 생성하였고, 이를 csv파일로 저장하였습니다. R 에서 제공하는 다양한 반복 처리 기법들을 사용하여 생성한 csv파일을 각각 500번씩 읽어들이는데 소요된 속도를 `system.time()` 함수를 사용하여 비교해보도록 하겠습니다.


```r
# 테스트 결과 리스트에 저장
speed_result <- list()
speed_result <- list(
    # for문
    forLoop = system.time({
        for(i in 1:500){
            read.csv('testFile.csv')
        }
    }),

    # lapply문
    lapply = system.time({
        lapply(1:500, function(x){
            read.csv('testFile.csv')
        })
    }),

    # sapply문
    sapply = system.time({
        sapply(1:500, function(x){
            read.csv('testFile.csv')
        })
    }),

    # purrr::map문
    map = system.time({
        purrr::map(1:500, function(x){
            read.csv('testFile.csv')
        })
    }),

    # future문
    future = system.time({
        future({
            lapply(1:500, function(x){
                read.csv('testFile.csv')
            })
        }) %>% value()
    })
)

speed_result
```

```
#> $forLoop
#>  사용자  시스템 elapsed 
#>   20.42    0.47   20.97 
#> 
#> $lapply
#>  사용자  시스템 elapsed 
#>   21.04    0.41   21.52 
#> 
#> $sapply
#>  사용자  시스템 elapsed 
#>   21.29    0.34   21.69 
#> 
#> $map
#>  사용자  시스템 elapsed 
#>   21.47    0.34   21.86 
#> 
#> $future
#>  사용자  시스템 elapsed 
#>    0.08    0.08   20.28
```

  **`future`** 패키지를 사용하였음에도 불구하고 예상한것과 달리 속도 개선이 그리 와닿지 않습니다. 다른 기법들과 속도가 비슷하거나 `purrr::map()` 함수와 비교해보면 되려 속도가 느린 것을 볼 수 있습니다. 여기서 한가지 더 짚고 넘어가야 할 점이라면 **`future`** 패키지를 통해 속도개선을 위한다면 다음의 패키지들을 추가적으로 설치하여 사용하는 것을 추천합니다.


```r
install.packages('future.apply')
install.packages('furrr')
```

  **`future`** 패키지의 확장 패키지들인 **`future.apply`** , **`furrr`** 패키지는 병렬 환경에서 각각 `apply` 문과 `purrr` 패키지의 함수를 대체하여 사용할 수 있는 `future.apply::future_apply()` 또는 `furrr::future_map()` 등의 함수를 제공하고 있습니다.


```r
# future_lapply 속도 측정 
library(future.apply)
list(
  future_apply = system.time({
    future_lapply(1:500, function(x){read.csv('testFile.csv')})
  })
)
```

```
#> $future_apply
#>  사용자  시스템 elapsed 
#>    0.22    0.20    8.53
```

```r
# future_map 속도 측정 
library(furrr)
list(
  future_map = system.time({
    future_map(1:500, function(x){read.csv('testFile.csv')})
  })
)
```

```
#> $future_map
#>  사용자  시스템 elapsed 
#>    0.28    0.16    7.72
```

 `future()` 함수의 속도 대비 훨신 빠른 성능을 보인 것을 확인할 수 있습니다.

#### 2-2) 모델링 속도 비교

 이번에는 실제 모델링에서의 연산속도는 어떠할지 측정해 보도록 하겠습니다. 시계열분석 예제 데이터로 많이 활용되는 `AirPassengers` 데이터에 난수를 더하여 만든 50개의 시계열 데이터에 각각 ARIMA모형을 구축하고, 각 방법마다 모델링에 걸리는 시간을 측정 및 비교해 보도록 하겠습니다. 이번에도 역시 다른 일반적인 기법들과의 비교를 통해 속도 측정을 진행해 보도록 하죠.


```r
library(forecast)

# 속도 측정용 데이터 생성 
test_range <- 1:50
testdata <- lapply(test_range, function(x){
    AirPassengers + sample(144)
}) %>%
    as.data.frame() %>%
    rename_with(~paste0('air', test_range))

glimpse(testdata)
```

```
#> Rows: 144
#> Columns: 50
#> $ air1  <dbl> 132, 217, 261, 156, 224, 242, 196, 233, 277, 177, 226, 133, 253,~
#> $ air2  <dbl> 235, 182, 168, 159, 203, 142, 191, 157, 146, 206, 185, 137, 131,~
#> $ air3  <dbl> 181, 140, 156, 218, 244, 263, 149, 259, 186, 237, 121, 257, 252,~
#> $ air4  <dbl> 160, 122, 146, 211, 164, 266, 180, 228, 139, 200, 141, 251, 172,~
#> $ air5  <dbl> 123, 241, 219, 204, 211, 277, 189, 211, 244, 123, 146, 194, 203,~
#> $ air6  <dbl> 251, 190, 265, 150, 187, 180, 255, 274, 276, 126, 138, 182, 207,~
#> $ air7  <dbl> 246, 211, 133, 175, 123, 188, 191, 255, 275, 183, 198, 241, 185,~
#> $ air8  <dbl> 199, 152, 135, 204, 245, 171, 171, 222, 188, 246, 144, 120, 175,~
#> $ air9  <dbl> 148, 198, 218, 161, 187, 158, 156, 208, 266, 126, 154, 132, 199,~
#> $ air10 <dbl> 209, 235, 160, 148, 137, 174, 261, 202, 171, 231, 191, 200, 206,~
#> $ air11 <dbl> 153, 148, 255, 221, 241, 276, 156, 170, 229, 247, 138, 217, 237,~
#> $ air12 <dbl> 171, 240, 180, 131, 245, 249, 227, 190, 162, 137, 121, 146, 179,~
#> $ air13 <dbl> 204, 121, 232, 198, 246, 256, 257, 274, 179, 172, 194, 230, 209,~
#> $ air14 <dbl> 180, 189, 271, 172, 185, 236, 258, 163, 144, 145, 170, 221, 162,~
#> $ air15 <dbl> 252, 191, 233, 149, 160, 263, 163, 175, 242, 130, 192, 159, 224,~
#> $ air16 <dbl> 162, 214, 247, 250, 174, 244, 205, 206, 259, 196, 211, 237, 197,~
#> $ air17 <dbl> 119, 246, 137, 142, 241, 246, 191, 174, 157, 171, 158, 179, 140,~
#> $ air18 <dbl> 122, 254, 166, 140, 245, 238, 254, 167, 153, 227, 135, 138, 243,~
#> $ air19 <dbl> 236, 190, 258, 230, 230, 180, 195, 245, 191, 230, 145, 231, 209,~
#> $ air20 <dbl> 136, 223, 246, 156, 169, 227, 285, 244, 178, 177, 159, 134, 175,~
#> $ air21 <dbl> 168, 245, 149, 164, 154, 140, 175, 259, 200, 149, 242, 241, 160,~
#> $ air22 <dbl> 114, 183, 212, 273, 199, 265, 217, 210, 248, 179, 133, 125, 166,~
#> $ air23 <dbl> 175, 253, 218, 146, 149, 155, 237, 166, 175, 253, 230, 159, 248,~
#> $ air24 <dbl> 247, 212, 238, 269, 177, 179, 250, 189, 225, 260, 109, 211, 202,~
#> $ air25 <dbl> 148, 200, 186, 246, 246, 138, 176, 220, 215, 252, 138, 167, 227,~
#> $ air26 <dbl> 125, 223, 215, 164, 231, 190, 151, 225, 199, 244, 129, 158, 162,~
#> $ air27 <dbl> 149, 237, 133, 133, 230, 137, 261, 232, 252, 190, 233, 144, 130,~
#> $ air28 <dbl> 142, 208, 227, 207, 245, 156, 219, 255, 177, 127, 209, 192, 183,~
#> $ air29 <dbl> 177, 221, 141, 156, 188, 207, 279, 186, 210, 141, 221, 223, 231,~
#> $ air30 <dbl> 135, 179, 210, 244, 179, 251, 290, 216, 270, 248, 145, 142, 178,~
#> $ air31 <dbl> 116, 195, 276, 134, 134, 144, 230, 201, 230, 169, 238, 147, 155,~
#> $ air32 <dbl> 232, 171, 267, 239, 144, 275, 179, 149, 186, 244, 156, 245, 143,~
#> $ air33 <dbl> 255, 228, 233, 218, 206, 168, 235, 243, 190, 230, 219, 180, 168,~
#> $ air34 <dbl> 187, 260, 233, 239, 265, 144, 161, 282, 191, 236, 217, 255, 206,~
#> $ air35 <dbl> 211, 137, 136, 186, 137, 204, 153, 161, 236, 212, 132, 246, 199,~
#> $ air36 <dbl> 145, 212, 224, 159, 135, 272, 221, 184, 280, 171, 135, 224, 199,~
#> $ air37 <dbl> 235, 214, 265, 212, 172, 245, 264, 220, 215, 169, 211, 188, 132,~
#> $ air38 <dbl> 229, 125, 182, 239, 224, 236, 273, 219, 144, 137, 141, 141, 129,~
#> $ air39 <dbl> 178, 139, 193, 183, 224, 191, 256, 150, 164, 251, 116, 215, 128,~
#> $ air40 <dbl> 122, 141, 133, 207, 185, 227, 206, 277, 237, 174, 160, 212, 175,~
#> $ air41 <dbl> 244, 203, 207, 223, 219, 218, 244, 170, 197, 189, 140, 141, 248,~
#> $ air42 <dbl> 202, 120, 223, 251, 257, 278, 258, 289, 183, 205, 176, 119, 132,~
#> $ air43 <dbl> 190, 152, 233, 252, 129, 242, 264, 149, 250, 218, 129, 140, 158,~
#> $ air44 <dbl> 156, 230, 198, 268, 221, 255, 215, 232, 235, 123, 152, 260, 198,~
#> $ air45 <dbl> 253, 194, 138, 209, 251, 199, 180, 229, 180, 171, 169, 195, 197,~
#> $ air46 <dbl> 125, 253, 178, 256, 147, 172, 217, 213, 226, 127, 113, 206, 139,~
#> $ air47 <dbl> 250, 165, 165, 163, 137, 249, 219, 281, 177, 193, 198, 138, 188,~
#> $ air48 <dbl> 241, 261, 220, 252, 148, 255, 157, 183, 274, 234, 225, 172, 184,~
#> $ air49 <dbl> 188, 128, 184, 176, 135, 226, 225, 266, 264, 167, 241, 142, 254,~
#> $ air50 <dbl> 115, 164, 151, 199, 241, 204, 273, 209, 184, 181, 155, 186, 152,~
```

```r
# 시계열 데이터 변환 
testdata <- testdata %>% ts(frequency = 12, start = 1949)
```


```r
# 모델링 속도 비교 
modeling_speed_test <- list()
modeling_speed_test <- list(
    # For문
    forLoop = system.time({
        for(i in seq(1, ncol(testdata))){
            testdata[, i] %>% auto.arima()
        }
    }),
    
    # lapply문
    lapply = system.time({
        testdata %>% lapply(auto.arima)
    }),
    
    # sapply문
    sapply = system.time({
        testdata %>% sapply(auto.arima)
    }),
    
    # `future` For문
    future_forLoop = system.time({
        for(i in seq(1, ncol(testdata))){
            future({
                testdata[, i] %>% auto.arima()
            })
        }
    }),
    
    # future_lapply문
    future_lapply = system.time({
        testdata %>% future_lapply(auto.arima)
    }),
    
    # future_map문
    future_map = system.time({
        testdata %>% future_map(auto.arima)
    }), 
    
    #apply문 
    apply = system.time({
        testdata %>% apply(2, auto.arima)
    }),
    
    #future_apply문 
    future_apply = system.time({
        testdata %>% future_apply(2, auto.arima)
    })
)

modeling_speed_test
```

```
#> $forLoop
#>  사용자  시스템 elapsed 
#>  260.25    0.37  261.09 
#> 
#> $lapply
#>  사용자  시스템 elapsed 
#>  257.83    0.06  258.26 
#> 
#> $sapply
#>  사용자  시스템 elapsed 
#>  268.69    0.13  269.25 
#> 
#> $future_forLoop
#>  사용자  시스템 elapsed 
#>    0.98    0.01   96.67 
#> 
#> $future_lapply
#>  사용자  시스템 elapsed 
#>    1.41    0.02  113.04 
#> 
#> $future_map
#>  사용자  시스템 elapsed 
#>    0.64    0.02    4.21 
#> 
#> $apply
#>  사용자  시스템 elapsed 
#>    6.67    0.00    6.72 
#> 
#> $future_apply
#>  사용자  시스템 elapsed 
#>    0.47    0.01    2.31
```

 50번에 걸친 ARIMA모델링을 `sapply()` 함수가 269.25초 동안 처리한 것에 비해 `future_apply()` 함수는 2초 내외의 시간 안에 처리한 것을 확인할 수 있습니다. 이와 같이 **`future`** 패키지를 통한 비동기 및 병렬연산을 적절히 활용한다면 보다 빠르고 효율적인 R 프로그래밍을 진행 할 수 있습니다.

------------------------------------------------------------------------

# 참고자료

-   <https://github.com/HenrikBengtsson/future>

-   <https://github.com/HenrikBengtsson/future.apply>

-   <https://github.com/DavisVaughan/furrr>

-   [비동기 프로그래밍 WITH FUTURE (R-ADV WEEK1)](https://lumiamitie.github.io/r/radv-wk1-future/)

-   [My Keynote 'Future' Presentation at the European Bioconductor Meeting 2020](https://www.jottr.org/2020/12/19/future-eurobioc2020-slides/)

-   <https://cran.r-project.org/web/packages/promises/vignettes/futures.html>
