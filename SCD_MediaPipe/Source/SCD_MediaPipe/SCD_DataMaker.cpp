// Fill out your copyright notice in the Description page of Project Settings.


#include "SCD_DataMaker.h"

// Sets default values
ASCD_DataMaker::ASCD_DataMaker()
{


}

// Called when the game starts or when spawned
void ASCD_DataMaker::BeginPlay()
{
	Super::BeginPlay();
	
}

// Called every frame
void ASCD_DataMaker::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

}

void ASCD_DataMaker::SaveStringArrayToTxt(TArray<FString> Data)
{
	FString SaveDirectory = FPaths::ProjectSavedDir();
	FString FilePath = FPaths::Combine(*SaveDirectory, TEXT("MyFile.txt"));
	
	FFileHelper::SaveStringArrayToFile(Data, *FilePath);
}

