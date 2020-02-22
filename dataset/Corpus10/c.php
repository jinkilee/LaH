
<html>
<head>
<title>DB Insert</title>
<META HTTP-EQUIV='Content-Type' content="text/html; charset=utf-8">
<link rel="STYLESHEET" href="visualasp.css" type="text/css">
</head>
<body>

<?
// File Input
$get;

$file_name = $get;


	$file_content = file("./i_c/".trim($file_name)."");
	$file_content_c = file("./i_k/".trim($file_name)."");
	$file_pointer = fopen("./output_c/".$file_name."", "w"); 
	$count = 0;
	for($i=0; $i<count($file_content); $i++) {
		$temp = "";
		$temp = $file_content[$i];

		$temp_c = "";
		$temp_c = $file_content_c[$i];


		$count++;
		$key = $count%4;
		if($key == 1 || $key == 2  || $key == 3) {

			$text = $temp;
		}
		else {
			$text = $temp_c;	
		}
		echo $text . "<br>";
//		fwrite($file_pointer, $text);

	} 
	fclose($file_pointer); 
	echo $file_name. "<br>";


?> 

</body>
</html>