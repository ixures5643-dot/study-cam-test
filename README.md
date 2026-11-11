# 撮影判定テスト

スマホの前面カメラで、勉強している時間だけを数えられるかを試すための試作ページです。

- 人が映っているか（MediaPipe Pose Landmarker）
- 写真ではなく生きた人か＝体の小さな揺れ・画素の変化・まばたき（Pose / Face Landmarker）
- スマホを手に持っていないか（Object Detector・EfficientDet-Lite0）

映像は端末の中で処理するだけで、保存も送信もしません。

公開ページ: https://ixures5643-dot.github.io/study-cam-test/
