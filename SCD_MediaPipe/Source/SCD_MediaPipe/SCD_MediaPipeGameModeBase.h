// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/GameModeBase.h"
#include "SCD_MediaPipeGameModeBase.generated.h"

/**
 * 
 */
UCLASS()
class SCD_MEDIAPIPE_API ASCD_MediaPipeGameModeBase : public AGameModeBase
{
	GENERATED_BODY()
public:
	ASCD_MediaPipeGameModeBase();
	
	void Tick(float DeltaSeconds) override;
	void BeginPlay() override; 

	UFUNCTION()
	void ReceiveFloatArrayFromPython();

	
};
