param(
  [Parameter(Mandatory=$true)][string]$MetricsPath,
  [Parameter(Mandatory=$true)][string]$OutputDir
)

Add-Type -AssemblyName System.Drawing -ErrorAction Stop
$data = Get-Content -Raw -LiteralPath $MetricsPath | ConvertFrom-Json -ErrorAction Stop
$null = New-Item -ItemType Directory -Path $OutputDir -Force

function Draw-GroupedChart {
  param([string]$Title,[string[]]$Labels,[string[]]$Series,[double[][]]$Values,[string]$FilePath)
  $width=1100; $height=650; $left=100; $top=90; $plotWidth=930; $plotHeight=420; $bottom=$top+$plotHeight
  $bitmap=[System.Drawing.Bitmap]::new($width,$height)
  $graphics=[System.Drawing.Graphics]::FromImage($bitmap)
  $graphics.SmoothingMode=[System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
  $graphics.Clear([System.Drawing.Color]::White)
  $fontTitle=[System.Drawing.Font]::new('Arial',20,[System.Drawing.FontStyle]::Bold)
  $font=[System.Drawing.Font]::new('Arial',11)
  $fontSmall=[System.Drawing.Font]::new('Arial',9)
  $black=[System.Drawing.Brushes]::Black
  $gray=[System.Drawing.Pen]::new([System.Drawing.Color]::FromArgb(220,220,220),1)
  $axis=[System.Drawing.Pen]::new([System.Drawing.Color]::FromArgb(80,80,80),1.5)
  $colors=@(
    [System.Drawing.Color]::FromArgb(43,112,173),
    [System.Drawing.Color]::FromArgb(232,135,48),
    [System.Drawing.Color]::FromArgb(70,150,95)
  )
  $graphics.DrawString($Title,$fontTitle,$black,100,25)
  for($tick=0;$tick -le 5;$tick++){
    $fraction=$tick/5.0; $y=$bottom-$fraction*$plotHeight
    $graphics.DrawLine($gray,$left,[single]$y,$left+$plotWidth,[single]$y)
    $graphics.DrawString(([string]::Format('{0:0.0}',$fraction)),$fontSmall,$black,55,[single]($y-8))
  }
  $graphics.DrawLine($axis,$left,$top,$left,$bottom)
  $graphics.DrawLine($axis,$left,$bottom,$left+$plotWidth,$bottom)
  $groupWidth=$plotWidth/[Math]::Max(1,$Labels.Count)
  $barWidth=[Math]::Min(55,$groupWidth/([Math]::Max(1,$Series.Count)+1))
  for($g=0;$g -lt $Labels.Count;$g++){
    $groupCenter=$left+($g+0.5)*$groupWidth
    for($s=0;$s -lt $Series.Count;$s++){
      $value=[Math]::Max(0.0,[Math]::Min(1.0,[double]$Values[$g][$s]))
      $barX=$groupCenter+($s-($Series.Count-1)/2.0)*($barWidth+4)-$barWidth/2
      $barHeight=$value*$plotHeight; $barY=$bottom-$barHeight
      $brush=[System.Drawing.SolidBrush]::new($colors[$s % $colors.Count])
      $graphics.FillRectangle($brush,[single]$barX,[single]$barY,[single]$barWidth,[single]$barHeight)
      $brush.Dispose()
    }
    $label=$Labels[$g]
    $size=$graphics.MeasureString($label,$fontSmall)
    $graphics.DrawString($label,$fontSmall,$black,[single]($groupCenter-$size.Width/2),[single]($bottom+12))
  }
  $legendY=565; $legendX=150
  for($s=0;$s -lt $Series.Count;$s++){
    $brush=[System.Drawing.SolidBrush]::new($colors[$s % $colors.Count])
    $graphics.FillRectangle($brush,$legendX,$legendY,18,14)
    $graphics.DrawString($Series[$s],$font,$black,$legendX+25,$legendY-3)
    $legendX += 270
    $brush.Dispose()
  }
  $bitmap.Save($FilePath,[System.Drawing.Imaging.ImageFormat]::Png)
  $gray.Dispose(); $axis.Dispose(); $fontTitle.Dispose(); $font.Dispose(); $fontSmall.Dispose(); $graphics.Dispose(); $bitmap.Dispose()
}

$algorithms=@('Nearest Road','HMM/Viterbi','Adaptive Radius')
$overall=@($data.test_results.overall)
$overallLabels=@('Nearest Road','HMM/Viterbi','Adaptive Radius')
$overallSeries=@('Edge accuracy','Macro F1')
$overallValues=@()
foreach($name in $algorithms){
  $r=$overall|Where-Object algorithm -eq $name|Select-Object -First 1
  $overallValues+=,@([double]$r.metrics.edge_accuracy,[double]$r.metrics.edge_f1_macro)
}
Draw-GroupedChart -Title 'Overall test-set comparison' -Labels $overallLabels -Series $overallSeries -Values $overallValues -FilePath (Join-Path $OutputDir 'algorithm_comparison.png')

$contexts=@('open_highway','urban_canyon','complex_intersection','airport_multilevel')
$contextValues=@()
foreach($context in $contexts){
  $row=@(); foreach($name in $algorithms){$r=@($data.test_results.by_context|Where-Object {$_.context -eq $context -and $_.algorithm -eq $name})[0];$row+=[double]$r.metrics.edge_accuracy};$contextValues+=,@($row)
}
Draw-GroupedChart -Title 'Test accuracy by road context' -Labels $contexts -Series $algorithms -Values $contextValues -FilePath (Join-Path $OutputDir 'context_wise_comparison.png')

$intervals=@(1,5,15,30,45,60,120)
$intervalLabels=@($intervals|ForEach-Object {"${_}s"})
$intervalValues=@()
foreach($interval in $intervals){
  $row=@(); foreach($name in $algorithms){$r=@($data.test_results.by_sampling_interval|Where-Object {[int]$_.sampling_interval_s -eq $interval -and $_.algorithm -eq $name})[0];$row+=[double]$r.metrics.edge_accuracy};$intervalValues+=,@($row)
}
Draw-GroupedChart -Title 'Test accuracy by sampling interval' -Labels $intervalLabels -Series $algorithms -Values $intervalValues -FilePath (Join-Path $OutputDir 'sampling_interval_comparison.png')

Get-ChildItem -LiteralPath $OutputDir -Filter '*.png' | ForEach-Object { Write-Output ("Created " + $_.FullName) }

