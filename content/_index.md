---
title: JDW Blog
type: landing
design:
  spacing: "5rem"

sections:
  - block: resume-biography
    id: about
    content:
      username: jdw
    design:
      spacing:
        padding: ["3rem", 0, "2rem", 0]
      avatar:
        size: large
        shape: circle

  - block: embedded-app
    id: interactive-profile
    content:
      title: "Interactive R"
      text: "브라우저에서 실행되는 독립형 Shinylive/webR 앱입니다."
      app_url: "/apps/profile-widget/"
      app_title: "JDW Interactive R App"
      height: 500
      loading: lazy
    design:
      spacing:
        padding: ["2rem", 0, "4rem", 0]

  - block: collection
    id: posts
    content:
      title: "최근 게시물"
      text: "R, Python, 클라우드와 데이터 제품에 관한 기록"
      count: 6
      filters:
        folders:
          - post
    design:
      view: card
      spacing:
        padding: ["4rem", 0, "4rem", 0]

  - block: collection
    id: projects
    content:
      title: "프로젝트"
      count: 6
      filters:
        folders:
          - project
    design:
      view: card
      spacing:
        padding: ["4rem", 0, "4rem", 0]

  - block: cta-button-list
    content:
      title: "Study Notes"
      text: "Kaggle과 딥러닝 학습 기록을 살펴보세요."
      buttons:
        - text: "Study Notes 보기"
          url: "/courses/"
          icon: hero/book-open
    design:
      spacing:
        padding: ["4rem", 0, "5rem", 0]
---
