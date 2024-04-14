// Copyright Epic Games, Inc. All Rights Reserved.


#include "SCD_MediaPipeGameModeBase.h"

#include "Networking.h"

ASCD_MediaPipeGameModeBase::ASCD_MediaPipeGameModeBase()
{
	PrimaryActorTick.bCanEverTick = true;
}

void ASCD_MediaPipeGameModeBase::Tick(float DeltaSeconds)
{
	Super::Tick(DeltaSeconds);
	ReceiveFloatArrayFromPython();
}

void ASCD_MediaPipeGameModeBase::BeginPlay()
{
	Super::BeginPlay();
	
	
}


void ASCD_MediaPipeGameModeBase::ReceiveFloatArrayFromPython()
{
	
	
	// 소켓 생성
	FSocket* Socket = ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM)->CreateSocket(NAME_Stream, TEXT("default"), false);
    
	// 서버 주소 설정
	TSharedRef<FInternetAddr> Address = ISocketSubsystem::Get(PLATFORM_SOCKETSUBSYSTEM)->CreateInternetAddr();
	bool bIsValid;
	Address->SetIp(TEXT("127.0.0.1"), bIsValid);
	Address->SetPort(12345);
	
	// 서버에 연결
	bool bConnected = Socket->Connect(*Address);
	if (!bConnected)
	{
		UE_LOG(LogTemp, Error, TEXT("Failed to connect to the server."));
		return;
	}
	
	/*float ReceivedFloatData1 = 0, ReceivedFloatData2 = 0, ReceivedFloatData3 = 0, ReceivedFloatData4, ReceivedFloatData5, ReceivedFloatData6, ReceivedFloatData7;
	
	Socket->Recv(reinterpret_cast<uint8*>(&ReceivedFloatData1), sizeof(float), BytesRead);
	Socket->Recv(reinterpret_cast<uint8*>(&ReceivedFloatData2), sizeof(float), BytesRead);
	Socket->Recv(reinterpret_cast<uint8*>(&ReceivedFloatData3), sizeof(float), BytesRead);
	Socket->Recv(reinterpret_cast<uint8*>(&ReceivedFloatData4), sizeof(float), BytesRead);
	Socket->Recv(reinterpret_cast<uint8*>(&ReceivedFloatData5), sizeof(float), BytesRead);
	Socket->Recv(reinterpret_cast<uint8*>(&ReceivedFloatData6), sizeof(float), BytesRead);
	Socket->Recv(reinterpret_cast<uint8*>(&ReceivedFloatData7), sizeof(float), BytesRead);*/

	int32 BytesRead = 0;
	char ReceivedStringBuffer[448]; // 적절한 크기로 설정
	Socket->Recv(reinterpret_cast<uint8*>(ReceivedStringBuffer), 448, BytesRead);

	TArray<float> ReceivedFloatDatas;
	FString ReceivedStringElement;
	for(auto& ch : ReceivedStringBuffer)
	{
		if(ch == ' ')
		{
			// 문자열을 FLOAT 값으로 변환
			ReceivedFloatDatas.Add(FCString::Atof(*ReceivedStringElement));
			ReceivedStringElement = TEXT("");
		}
		else
		{
			ReceivedStringElement += ch;
		}
	}
	

	for(int i = 0; i < ReceivedFloatDatas.Num(); i++)
	{
		float f = ReceivedFloatDatas[i];
		UE_LOG(LogTemp, Warning, TEXT("Received float value: %f"), f);
	}
	UE_LOG(LogTemp, Warning, TEXT("-------------------------------------"));
	// 소켓 닫기
	Socket->Close();

	/*// 데이터 수신
	//uint32 Size;
	/*TArray<uint8> ReceivedData;
	UE_LOG(LogTemp, Warning, TEXT("  scsdfsdfsd %d"), Size);
	while (Socket->HasPendingData(Size))
	{
		ReceivedData.Init(0, FMath::Min(Size, 65507u));
		int32 Read = 0;
		Socket->Recv(ReceivedData.GetData(), ReceivedData.Num(), Read);
	}#1#
	TArray<uint8> ReceivedData;
	
	ReceivedData.Reset();
	
	if (Socket->Wait(ESocketWaitConditions::WaitForRead, FTimespan::FromSeconds(1)))
	{
		
		ReceivedData.SetNumUninitialized(MAX_PACKET_SIZE);
		if (Socket->Recv(ReceivedData.GetData(), ReceivedData.Num(), BytesRead))
		{
			ReceivedData.SetNum(BytesRead);
			// 데이터 처리
			// 수신한 데이터를 float 배열로 변환
			TArray<float> FloatArray;
			FloatArray.SetNum(ReceivedData.Num() / sizeof(float));
			FMemory::Memcpy(FloatArray.GetData(), ReceivedData.GetData(), ReceivedData.Num());
			
			
			
			/#1#/ 수신한 데이터 출력
			for (float Value : FloatArray)
			{
				UE_LOG(LogTemp, Warning, TEXT("Received float value: %f"), Value);
			}#1#
		}
	}
	


	*/
	

	
}
